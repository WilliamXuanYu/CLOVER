"""Stage-2 DrivoR fine-tuning with optional teacher-guided phases."""

from typing import Any, Dict
import copy
import logging
import math

import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

from .drivor_agent import DrivoRAgent, LitProgressBar

logger = logging.getLogger(__name__)


class TeacherRefreshCallback(Callback):
    """Periodically refresh the frozen teacher from the current student."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is None or not hasattr(agent, "maybe_refresh_teacher"):
            return
        agent.maybe_refresh_teacher(int(trainer.current_epoch))


class AlternatingPhaseCallback(Callback):
    """Switch critic/generator phase inside one Trainer run."""

    def on_fit_start(self, trainer, pl_module) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is None or not hasattr(agent, "set_training_phase_for_epoch"):
            return
        agent.set_training_phase_for_epoch(0, reason="fit_start")

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is None or not hasattr(agent, "set_training_phase_for_epoch"):
            return
        agent.set_training_phase_for_epoch(int(trainer.current_epoch), reason="epoch_start")


class EMATeacherUpdateCallback(Callback):
    """Update the teacher with an EMA of the student after train batches."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        agent = getattr(pl_module, "agent", None)
        if agent is None or not hasattr(agent, "maybe_update_teacher_ema"):
            return
        agent.maybe_update_teacher_ema(int(trainer.global_step))


class DrivoRStage2Agent(DrivoRAgent):
    def __init__(
        self,
        config,
        lr_args: dict,
        checkpoint_path: str = "",
        base_checkpoint_path: str = "",
        loss=None,
        progress_bar: bool = True,
        scheduler_args: dict = None,
        batch_size: int = 64,
        num_gpus: int = 1,
        trajectory_sampling: Any = None,
    ):
        self._base_checkpoint_path = base_checkpoint_path
        super().__init__(
            config=config,
            lr_args=lr_args,
            checkpoint_path=checkpoint_path,
            loss=loss,
            progress_bar=progress_bar,
            scheduler_args=scheduler_args,
            batch_size=batch_size,
            num_gpus=num_gpus,
            trajectory_sampling=trajectory_sampling,
        )

        if self._base_checkpoint_path and not self._checkpoint_path:
            self._load_base_checkpoint(self._base_checkpoint_path)

        self._teacher_model = copy.deepcopy(self._drivor_model)
        self._teacher_refresh_interval = int(getattr(self._config, "teacher_refresh_interval", 5))
        self._last_teacher_refresh_epoch = -1
        self._use_ema_teacher = bool(getattr(self._config, "use_ema_teacher_in_stage2", False))
        self._ema_teacher_momentum = float(getattr(self._config, "ema_teacher_momentum", 0.995))
        self._ema_teacher_update_interval = max(
            1, int(getattr(self._config, "ema_teacher_update_interval", 1))
        )
        self._alternating_stage2 = bool(getattr(self._config, "alternating_stage2", False))
        self._critic_phase_epochs = max(1, int(getattr(self._config, "critic_phase_epochs", 1)))
        self._generator_phase_epochs = max(1, int(getattr(self._config, "generator_phase_epochs", 1)))
        self._initial_stage2_phase = str(getattr(self._config, "stage2_training_phase", "generator")).lower()
        self._current_stage2_phase = self._initial_stage2_phase
        self._freeze_module(self._teacher_model)
        self._teacher_model.eval()

        self._apply_training_phase_freeze()

        self._sync_teacher_from_student(reason="init")

    def name(self) -> str:
        return "DrivoRStage2Agent"

    def _load_base_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        mapped: Dict[str, torch.Tensor] = {}
        has_refine_scorer = False
        for k, v in state_dict.items():
            if "_drivor_model" in k:
                mapped[k.replace("agent._drivor_model", "_drivor_model")] = v
                if "refine_scorer" in k:
                    has_refine_scorer = True
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        if (
            bool(getattr(self._config, "use_refine_scorer", False))
            and bool(getattr(self._config, "init_refine_scorer_from_base", True))
            and not has_refine_scorer
            and hasattr(self._drivor_model, "initialize_refine_scorer_from_base")
        ):
            self._drivor_model.initialize_refine_scorer_from_base()
            logger.info("Initialized refine scorer from base scorer after loading %s.", path)
        logger.info(
            "Loaded %d base-model params from %s (missing=%d, unexpected=%d)",
            len(mapped),
            path,
            len(missing),
            len(unexpected),
        )

    def _freeze_module(self, module) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = True

    def _unfreeze_student_modules(self) -> None:
        for param in self._drivor_model.parameters():
            param.requires_grad = True

    def _freeze_scorer_modules(self) -> None:
        self._freeze_module(getattr(self._drivor_model, "pos_embed", None))
        self._freeze_module(getattr(self._drivor_model, "scorer_attention", None))
        self._freeze_module(getattr(self._drivor_model, "scorer", None))
        self._freeze_module(getattr(self._drivor_model, "rc_encoder", None))
        logger.info("Stage-2 scorer-related modules are frozen.")

    def _freeze_refine_scorer_modules(self) -> None:
        self._freeze_module(getattr(self._drivor_model, "refine_pos_embed", None))
        self._freeze_module(getattr(self._drivor_model, "refine_scorer_attention", None))
        self._freeze_module(getattr(self._drivor_model, "refine_scorer", None))
        logger.info("Stage-2 refine-scorer modules are frozen.")

    def _unfreeze_refine_scorer_modules(self) -> None:
        self._unfreeze_module(getattr(self._drivor_model, "refine_pos_embed", None))
        self._unfreeze_module(getattr(self._drivor_model, "refine_scorer_attention", None))
        self._unfreeze_module(getattr(self._drivor_model, "refine_scorer", None))
        logger.info("Stage-2 refine-scorer modules are trainable.")

    def _freeze_backbone_modules(self) -> None:
        self._freeze_module(getattr(self._drivor_model, "image_backbone", None))
        self._freeze_module(getattr(self._drivor_model, "lidar_backbone", None))
        logger.info("Stage-2 perception backbones are frozen.")

    def _freeze_generator_modules(self) -> None:
        self._freeze_module(getattr(self._drivor_model, "hist_encoding", None))
        self._freeze_module(getattr(self._drivor_model, "init_feature", None))
        self._freeze_module(getattr(self._drivor_model, "trajectory_decoder", None))
        self._freeze_module(getattr(self._drivor_model, "traj_head", None))
        logger.info("Stage-2 generator-related modules are frozen.")

    def _apply_training_phase_freeze(self) -> None:
        self._unfreeze_student_modules()
        phase = str(getattr(self._config, "stage2_training_phase", "joint")).lower()
        if phase == "critic":
            self._freeze_generator_modules()
        elif phase == "generator":
            self._freeze_scorer_modules()

        if bool(getattr(self._config, "freeze_scorer_in_stage2", False)):
            self._freeze_scorer_modules()
        if bool(getattr(self._config, "freeze_backbone_in_stage2", False)):
            self._freeze_backbone_modules()
        if bool(getattr(self._config, "train_refinement_alternating", False)):
            self._freeze_module(self._drivor_model)
            if phase == "critic":
                self._unfreeze_refine_scorer_modules()
                logger.info("Only D2 refine scorer is trainable.")
            else:
                self._unfreeze_module(getattr(self._drivor_model, "refinement_head", None))
                logger.info("Only D2 refinement head is trainable.")
            return
        if bool(getattr(self._config, "train_refinement_head_only", False)):
            self._freeze_module(self._drivor_model)
            self._unfreeze_module(getattr(self._drivor_model, "refinement_head", None))
            logger.info("Only stage-2 refinement head is trainable.")

    def _sync_teacher_from_student(self, reason: str = "manual") -> None:
        self._teacher_model.load_state_dict(self._drivor_model.state_dict(), strict=True)
        self._freeze_module(self._teacher_model)
        self._teacher_model.eval()
        logger.info("Refreshed stage-2 teacher from student (%s).", reason)

    @torch.no_grad()
    def _update_teacher_from_student_ema(self) -> None:
        momentum = min(max(self._ema_teacher_momentum, 0.0), 0.99999)
        student_state = self._drivor_model.state_dict()
        teacher_state = self._teacher_model.state_dict()
        for key, teacher_value in teacher_state.items():
            student_value = student_state[key]
            if teacher_value.dtype.is_floating_point:
                teacher_value.mul_(momentum).add_(student_value.detach(), alpha=1.0 - momentum)
            else:
                teacher_value.copy_(student_value)
        self._freeze_module(self._teacher_model)
        self._teacher_model.eval()

    def _phase_for_epoch(self, current_epoch: int) -> str:
        if not self._alternating_stage2:
            return self._initial_stage2_phase

        cycle_len = self._critic_phase_epochs + self._generator_phase_epochs
        offset = current_epoch % cycle_len
        if self._initial_stage2_phase == "generator":
            if offset < self._generator_phase_epochs:
                return "generator"
            return "critic"

        if offset < self._critic_phase_epochs:
            return "critic"
        return "generator"

    def set_training_phase(self, phase: str, reason: str = "manual") -> None:
        phase = str(phase).lower()
        if phase == self._current_stage2_phase and reason != "fit_start_0":
            return
        self._current_stage2_phase = phase
        self._config.stage2_training_phase = phase
        if self._alternating_stage2:
            self._config.detach_proposals_in_scorer = phase == "critic"
        if hasattr(self.loss, "stage2_training_phase"):
            self.loss.stage2_training_phase = phase
        self._apply_training_phase_freeze()
        if phase == "generator" and not self._use_ema_teacher:
            self._sync_teacher_from_student(reason=f"{reason}_to_generator")
        logger.info("Stage-2 phase set to %s (%s).", phase, reason)

    def set_training_phase_for_epoch(self, current_epoch: int, reason: str = "epoch") -> None:
        self.set_training_phase(self._phase_for_epoch(current_epoch), reason=f"{reason}_{current_epoch}")

    def maybe_refresh_teacher(self, current_epoch: int) -> None:
        if self._use_ema_teacher:
            return
        if self._teacher_refresh_interval <= 0:
            return
        if current_epoch <= 0:
            return
        if current_epoch % self._teacher_refresh_interval != 0:
            return
        if self._last_teacher_refresh_epoch == current_epoch:
            return
        self._sync_teacher_from_student(reason=f"epoch_{current_epoch}")
        self._last_teacher_refresh_epoch = current_epoch

    def maybe_update_teacher_ema(self, global_step: int) -> None:
        if not self._use_ema_teacher:
            return
        if global_step < 0:
            return
        if global_step % self._ema_teacher_update_interval != 0:
            return
        self._update_teacher_from_student_ema()

    def initialize(self) -> None:
        super().initialize()
        self._sync_teacher_from_student(reason="initialize")

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        student_output = self._drivor_model(features)
        if bool(getattr(self._config, "disable_teacher_forward_in_stage2", False)):
            return student_output
        with torch.no_grad():
            self._teacher_model.eval()
            teacher_output = self._teacher_model(features)
        student_output["teacher_proposals"] = teacher_output["proposals"].detach()
        if "pdm_score" in teacher_output:
            student_output["teacher_pdm_score"] = teacher_output["pdm_score"].detach()
        if "pred_logit" in teacher_output:
            student_output["teacher_pred_logit"] = {
                key: value.detach() for key, value in teacher_output["pred_logit"].items()
            }
        return student_output

    def get_training_callbacks(self):
        # Stage-2 skips legacy val/score (on_step+on_epoch) in AgentLightningModule,
        # so val/score_epoch is never logged; val/top1_score is the same quantity.
        checkpoint_cb_best = ModelCheckpoint(
            save_top_k=1,
            monitor="val/top1_score",
            filename="best-{epoch}-{step}",
            mode="max",
        )
        checkpoint_cb_last = ModelCheckpoint(save_last=True)
        lr_monitor = LearningRateMonitor(
            logging_interval="step",
            log_momentum=False,
            log_weight_decay=False,
        )
        # Do not register a ProgressBar callback here: the trainer is launched
        # with enable_progress_bar=false in our stage-2 scripts, and Lightning
        # treats any custom ProgressBar callback as a conflicting configuration.
        callbacks = [checkpoint_cb_best, checkpoint_cb_last, lr_monitor, TeacherRefreshCallback()]
        if self._use_ema_teacher:
            callbacks.append(EMATeacherUpdateCallback())
        if self._alternating_stage2:
            callbacks.append(AlternatingPhaseCallback())
        return callbacks

    def get_optimizers(self):
        global_batchsize = self.batch_size * self.num_gpus
        lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        # In alternating mode, the active parameter subset changes across epochs.
        # Keep one optimizer over the whole student and rely on requires_grad flags
        # to decide which blocks receive updates in the current phase.
        params = list(self._drivor_model.parameters())
        if not params:
            raise RuntimeError("No trainable parameters left in DrivoRStage2Agent.")

        if self._lr_args["name"] == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        elif self._lr_args["name"] == "AdamW":
            optimizer = torch.optim.AdamW(params, lr=lr)
        else:
            raise NotImplementedError

        if self.scheduler_args is not None:
            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) * self.scheduler_args.num_epochs)
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine,
                eta_min=0.0,
                last_epoch=-1,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return [optimizer]
