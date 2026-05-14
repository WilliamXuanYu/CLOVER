from typing import Any, Dict, List, Union
import logging
import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivor_model import DrivoRModel
from .layers.image_encoder.dinov2_lora import _is_pytorch_lightning_checkpoint_path
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Scene, SensorConfig, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math

logger = logging.getLogger(__name__)


def _normalize_metric_cache_path_compat(path: str) -> str:
    """
    Keep metric-cache path normalization aligned with the shared loader.
    """
    normalize_fn = getattr(MetricCacheLoader, "_normalize_metric_cache_path", None)
    if callable(normalize_fn):
        return normalize_fn(str(path))

    path = str(path).strip().rstrip("\r")
    if len(path) >= 2 and path[0] == path[-1] == '"':
        path = path[1:-1].strip()
    if len(path) >= 2 and path[0] == path[-1] == "'":
        path = path[1:-1].strip()
    if os.path.exists(path):
        return path

    subst_from = os.environ.get("NAVSIM_METRIC_CACHE_PATH_SUBST_FROM", "").strip()
    subst_to = os.environ.get("NAVSIM_METRIC_CACHE_PATH_SUBST_TO", "").strip()
    if subst_from and subst_to and subst_from in path:
        substituted = path.replace(subst_from, subst_to, 1)
        if os.path.exists(substituted) or not os.path.exists(path):
            return substituted

    return path

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            trajectory_sampling: Any = None,
    ):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=False)
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self._inference_device = torch.device("cpu")


        cache_data=False

        if not cache_data:
            self._drivor_model = DrivoRModel(config)

        # compute_score can be used during evaluation when exporting real PDM scores.
        self.ray = bool(getattr(config, "use_ray_for_scoring", False))
        if self.ray:
            from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
            from nuplan.planning.utils.multithreading.worker_utils import worker_map

            requested_ray_threads = int(os.getenv("RAY_THREADS_PER_NODE", "8"))
            world_size = max(1, int(os.getenv("WORLD_SIZE", "1")))
            host_cpu_count = os.cpu_count() or requested_ray_threads
            reserved_cpu_headroom = max(0, int(os.getenv("RAY_CPU_HEADROOM", "8")))
            per_rank_cpu_budget = max(1, (host_cpu_count - reserved_cpu_headroom) // world_size)
            ray_threads = max(1, min(requested_ray_threads, per_rank_cpu_budget))
            if ray_threads < requested_ray_threads:
                logger.warning(
                    "Capping RAY_THREADS_PER_NODE from %d to %d "
                    "(WORLD_SIZE=%d, host_cpu_count=%d, reserved_cpu_headroom=%d) "
                    "to avoid CPU oversubscription during DDP training.",
                    requested_ray_threads,
                    ray_threads,
                    world_size,
                    host_cpu_count,
                    reserved_cpu_headroom,
                )
            self.worker = RayDistributedNoTorch(threads_per_node=ray_threads)
            self.worker_map = worker_map

        from .score_module.compute_navsim_score import get_scores
        self.get_scores = get_scores

        if not cache_data and self._checkpoint_path == "": # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            metric_cache_path = self._resolve_train_metric_cache_path()
            metric_cache = MetricCacheLoader(metric_cache_path)
            self.train_metric_cache_paths = metric_cache.metric_cache_paths

            self.loss = loss
            


    @staticmethod
    def _alternate_metric_cache_path(path: Path) -> Path:
        """Hook for downstream forks that want to customize cache path layouts."""
        return path

    def _resolve_train_metric_cache_path(self) -> Path:
        """
        Resolve metric cache root path and ensure metadata directory exists.
        Priority:
          1) NAVSIM_TRAIN_METRIC_CACHE_PATH
          2) NAVSIM_EXP_ROOT/train_metric_cache
        Uses an optional downstream override hook for path migration.
        """
        env_cache_path = os.getenv("NAVSIM_TRAIN_METRIC_CACHE_PATH")
        if env_cache_path:
            base_path = Path(env_cache_path)
        else:
            navsim_exp_root = os.getenv("NAVSIM_EXP_ROOT")
            if not navsim_exp_root:
                raise ValueError("NAVSIM_EXP_ROOT is not set.")
            base_path = Path(navsim_exp_root) / "train_metric_cache"

        candidates = [base_path]
        alternate = self._alternate_metric_cache_path(base_path)
        if alternate != base_path:
            candidates.append(alternate)

        for candidate in candidates:
            if (candidate / "metadata").is_dir():
                return candidate

        raise FileNotFoundError(
            "Metric cache metadata directory not found. Tried: "
            + ", ".join(str(path / "metadata") for path in candidates)
        )

    def _resolve_eval_metric_cache_path(self) -> Path:
        """
        Resolve metric cache root for evaluation-time score computation.
        Priority:
          1) NAVSIM_EVAL_METRIC_CACHE_PATH
          2) NAVSIM_TRAIN_METRIC_CACHE_PATH
          3) NAVSIM_EXP_ROOT/metric_cache
          4) NAVSIM_EXP_ROOT/train_metric_cache
        """
        candidates: List[Path] = []

        eval_cache_path = os.getenv("NAVSIM_EVAL_METRIC_CACHE_PATH")
        if eval_cache_path:
            candidates.append(Path(eval_cache_path))

        train_cache_path = os.getenv("NAVSIM_TRAIN_METRIC_CACHE_PATH")
        if train_cache_path:
            candidates.append(Path(train_cache_path))

        navsim_exp_root = os.getenv("NAVSIM_EXP_ROOT")
        if navsim_exp_root:
            candidates.append(Path(navsim_exp_root) / "metric_cache")
            candidates.append(Path(navsim_exp_root) / "train_metric_cache")

        expanded_candidates: List[Path] = []
        for candidate in candidates:
            expanded_candidates.append(candidate)
            alternate = self._alternate_metric_cache_path(candidate)
            if alternate != candidate:
                expanded_candidates.append(alternate)

        for candidate in expanded_candidates:
            if (candidate / "metadata").is_dir():
                return candidate

        raise FileNotFoundError(
            "Evaluation metric cache metadata directory not found. Tried: "
            + ", ".join(str(path / "metadata") for path in expanded_candidates)
        )

    def _ensure_metric_cache_paths(self, use_train_cache: bool) -> None:
        """Lazily initialize metric cache paths for compute_score in train/eval modes."""
        if use_train_cache:
            if hasattr(self, "train_metric_cache_paths"):
                return
            metric_cache = MetricCacheLoader(self._resolve_train_metric_cache_path())
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            return

        if hasattr(self, "test_metric_cache_paths") and self.test_metric_cache_paths is not None:
            return
        metric_cache = MetricCacheLoader(self._resolve_eval_metric_cache_path())
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        if not hasattr(self, "train_metric_cache_paths"):
            self.train_metric_cache_paths = metric_cache.metric_cache_paths

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path is None or self._checkpoint_path == "":
            return
        # Resolve path: strip quotes/whitespace and convert to absolute path
        raw = (self._checkpoint_path or "").strip().strip('"').strip("'")
        if not raw:
            return
        ckpt_path = Path(raw).resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_path}\n"
                f"Original path: {self._checkpoint_path!r}\n"
                f"Current working directory: {os.getcwd()}"
            )
        force_cpu_ckpt_load = os.getenv("DRIVOR_FORCE_CPU_CKPT_LOAD", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
        if torch.cuda.is_available() and not force_cpu_ckpt_load:
            state_dict: Dict[str, Any] = torch.load(str(ckpt_path))["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(str(ckpt_path), map_location=torch.device("cpu"))["state_dict"]

        # Only load the base DrivoRModel weights. This keeps evaluation compatible
        # with stage2 / reranker checkpoints that may also contain auxiliary heads
        # (e.g. teacher model, reranker modules) not present in DrivoRAgent.
        mapped_state_dict = {
            k.replace("agent._drivor_model", "_drivor_model"): v
            for k, v in state_dict.items()
            if k.startswith("agent._drivor_model.")
        }
        backbone_weights = None
        if hasattr(self._config, "image_backbone"):
            ib = self._config.image_backbone
            backbone_weights = (
                ib.get("model_weights", None) if hasattr(ib, "get") else getattr(ib, "model_weights", None)
            )
        mw = backbone_weights
        if mw is not None and not isinstance(mw, str):
            mw = str(mw)
        skip_backbone_from_init_ckpt = _is_pytorch_lightning_checkpoint_path((mw or "").strip())
        if skip_backbone_from_init_ckpt:
            logger.info(
                "Preserving image_backbone weights (image_backbone.model_weights is a PL checkpoint: %r); "
                "not loading _drivor_model.image_backbone.* from %s",
                backbone_weights,
                ckpt_path,
            )
        current_state_dict = self.state_dict()
        compatible_state_dict: Dict[str, Any] = {}
        skipped_missing = 0
        skipped_shape = 0
        for key, value in mapped_state_dict.items():
            if skip_backbone_from_init_ckpt and key.startswith("_drivor_model.image_backbone."):
                continue
            if key not in current_state_dict:
                skipped_missing += 1
                continue
            if current_state_dict[key].shape != value.shape:
                skipped_shape += 1
                logger.warning(
                    "Skipping checkpoint weight due to shape mismatch: %s (ckpt=%s, model=%s)",
                    key,
                    tuple(value.shape),
                    tuple(current_state_dict[key].shape),
                )
                continue
            compatible_state_dict[key] = value

        if not compatible_state_dict:
            raise RuntimeError(
                f"No compatible DrivoR weights found in checkpoint: {ckpt_path}"
            )

        missing, unexpected = self.load_state_dict(compatible_state_dict, strict=False)
        logger.info(
            "Loaded %d compatible checkpoint params from %s (skipped_missing=%d, skipped_shape=%d, missing_after_load=%d, unexpected_after_load=%d)",
            len(compatible_state_dict),
            ckpt_path,
            skipped_missing,
            skipped_shape,
            len(missing),
            len(unexpected),
        )
        self._inference_device = self._select_inference_device()
        self.to(self._inference_device)
        logger.info("DrivoR inference device: %s", self._inference_device)

    def _select_inference_device(self) -> torch.device:
        """Select a stable GPU device for multi-process evaluation workers."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        forced = os.getenv("DRIVOR_EVAL_DEVICE", "").strip()
        if forced:
            return torch.device(forced)

        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count <= 1:
            return torch.device("cuda:0")

        identity = multiprocessing.current_process()._identity
        if identity:
            gpu_index = (identity[0] - 1) % visible_gpu_count
        else:
            gpu_index = 0
        return torch.device(f"cuda:{gpu_index}")

    def _move_batch_to_device(self, data: Any) -> Any:
        if torch.is_tensor(data):
            return data.to(self._inference_device, non_blocking=True)
        if isinstance(data, dict):
            return {key: self._move_batch_to_device(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._move_batch_to_device(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self._move_batch_to_device(value) for value in data)
        return data

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        if bool(getattr(self._config, "refinement_cache_skip_sensors", False)) or (
            bool(getattr(self._config, "use_external_scene_features", False))
            and bool(getattr(self._config, "external_scene_feature_skip_sensors", True))
        ):
            return SensorConfig(
                cam_f0=[],
                cam_l0=[],
                cam_l1=[],
                cam_l2=[],
                cam_r0=[],
                cam_r1=[],
                cam_r2=[],
                cam_b0=[],
                lidar_pc=[],
            )
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._drivor_model(features)

    def compute_trajectory(self, agent_input) -> Trajectory:
        """Run single-scene inference on the configured device."""
        self.eval()

        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        batched_features = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        batched_features = self._move_batch_to_device(batched_features)

        with torch.no_grad():
            predictions = self.forward(batched_features)
            poses = predictions["trajectory"].squeeze(0).detach().cpu().numpy()

        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectory_with_details(
        self,
        agent_input,
        scene: Scene,
        save_all_proposals: bool = True,
        save_real_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Run single-scene inference and export the same per-token payload used by NAVSIM-v1 eval.
        """
        self.eval()

        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        token = scene.scene_metadata.initial_token
        batched_features: Dict[str, Any] = {}
        for key, value in features.items():
            if torch.is_tensor(value):
                batched_features[key] = value.unsqueeze(0)
            else:
                batched_features[key] = value
        batched_features.setdefault("scenario_token", [token])
        batched_features = self._move_batch_to_device(batched_features)

        with torch.no_grad():
            predictions = self.forward(batched_features)

        selected_pose = predictions["trajectory"][0].detach().cpu().numpy()
        result: Dict[str, Any] = {
            "trajectory": Trajectory(selected_pose, self._trajectory_sampling),
        }

        proposals = predictions.get("proposals")
        if proposals is not None and save_all_proposals:
            result["all_proposals"] = proposals[0].detach().cpu().numpy()

        scorer_scores = predictions.get("pdm_score")
        if scorer_scores is not None and (save_all_proposals or save_real_scores):
            result["scorer_scores"] = scorer_scores[0].detach().cpu().numpy()

        pred_logit = predictions.get("pred_logit", {})
        if pred_logit and (save_all_proposals or save_real_scores):
            result["scorer_subscores"] = {
                "no_at_fault_collisions": pred_logit["no_at_fault_collisions"][0].sigmoid().detach().cpu().numpy(),
                "drivable_area_compliance": pred_logit["drivable_area_compliance"][0].sigmoid().detach().cpu().numpy(),
                "driving_direction_compliance": pred_logit["driving_direction_compliance"][0].sigmoid().detach().cpu().numpy(),
                "time_to_collision_within_bound": pred_logit["time_to_collision_within_bound"][0].sigmoid().detach().cpu().numpy(),
                "ego_progress": pred_logit["ego_progress"][0].sigmoid().detach().cpu().numpy(),
                "comfort": pred_logit["comfort"][0].sigmoid().detach().cpu().numpy(),
            }

        if save_real_scores:
            if proposals is None:
                raise KeyError("Detailed export requires predictions['proposals'], but it is missing.")

            # Real PDM export only needs token->metric-cache lookup. A dummy trajectory
            # is sufficient here and avoids relying on GT future frames that may be
            # truncated in some NAVSIM-v2 synthetic stage-two scenes.
            batched_targets: Dict[str, Any] = {
                "token": [token],
                "trajectory": predictions["trajectory"].detach(),
            }
            _, _, final_scores, _, full_metrics = self.compute_score(
                batched_targets,
                proposals,
            )
            result["real_pdm_scores"] = final_scores[0].detach().cpu().numpy()
            result["all_proposal_scores"] = final_scores[0].detach().cpu().numpy()
            if full_metrics is not None:
                result["real_pdm_subscores"] = {
                    "no_at_fault_collisions": full_metrics[0, :, 0].detach().cpu().numpy(),
                    "drivable_area_compliance": full_metrics[0, :, 1].detach().cpu().numpy(),
                    "ego_progress": full_metrics[0, :, 2].detach().cpu().numpy(),
                    "time_to_collision_within_bound": full_metrics[0, :, 3].detach().cpu().numpy(),
                    "comfort": full_metrics[0, :, 4].detach().cpu().numpy(),
                    "driving_direction_compliance": full_metrics[0, :, 5].detach().cpu().numpy(),
                }

        ego_status = batched_features.get("ego_status")
        if torch.is_tensor(ego_status):
            result["high_level_command"] = ego_status[0].detach().cpu().numpy()

        return result

    def compute_score(self, targets, proposals, test=True, return_full_metrics=True):
        if self.training:
            self._ensure_metric_cache_paths(use_train_cache=True)
            metric_cache_paths = self.train_metric_cache_paths
        else:
            self._ensure_metric_cache_paths(use_train_cache=False)
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"]
        proposals = proposals.detach()
        if self.training:
            proposals = torch.nan_to_num(proposals, nan=0.0, posinf=100.0, neginf=-100.0)
            proposals_xy = proposals[..., :2].clamp(min=-100.0, max=100.0)
            proposals_heading = torch.atan2(torch.sin(proposals[..., 2]), torch.cos(proposals[..., 2]))
            proposals = torch.cat([proposals_xy, proposals_heading.unsqueeze(-1)], dim=-1)

        tokens = targets["token"]
        missing = [t for t in tokens if t not in metric_cache_paths]
        if missing:
            raise KeyError(
                f"Metric cache does not contain {len(missing)} token(s), e.g. '{missing[0]}'. "
                "Ensure the metric cache was built for this split (e.g. navtest). "
                "Set metric_cache_path in the script and NAVSIM_EVAL_METRIC_CACHE_PATH so the agent uses it, "
                "or run metric caching for navtest and point to that cache."
            ) from None

        data_points = [
            {
                "token": _normalize_metric_cache_path_compat(str(metric_cache_paths[token])),
                "poses": poses,
                "test": test,
                "return_full_metrics": return_full_metrics,
            }
            for token, poses in zip(tokens, proposals.cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        if test and not return_full_metrics:
            final_scores = torch.as_tensor(
                np.stack(all_res),
                dtype=torch.float32,
                device=proposals.device,
            )
            best_scores = torch.amax(final_scores, dim=-1)
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]
            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), None

        score_arrays = all_res if test else [res[0] for res in all_res]
        target_scores = torch.as_tensor(
            np.stack(score_arrays),
            dtype=torch.float32,
            device=proposals.device,
        )

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            # Return full per-proposal metric stacks [B, N, 7] so callers can index
            # the selected trajectory (not only proposal 0).
            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.Adam(self._drivor_model.parameters(), lr=lr)
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
            optimizer = torch.optim.AdamW(self._drivor_model.parameters(), lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           

            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max"
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        
        if self.progress_bar:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]
        else:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
