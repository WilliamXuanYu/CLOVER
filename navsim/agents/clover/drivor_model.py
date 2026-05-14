from typing import Dict
import copy
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .score_module.scorer import Scorer
from .transformer_decoder import Block, TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .region_consistency import RegionConsistencyEncoder
from navsim.agents.clover.utils import pylogger
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)


class SceneConditionedRefinementHead(nn.Module):
    """Small decoder that refines scorer-selected anchors with scene cross-attention."""

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        poses_num: int,
        state_size: int,
        score_dim: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        max_refinement_num: int = 16,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 0.0,
    ) -> None:
        super().__init__()
        self.poses_num = poses_num
        self.state_size = state_size
        self.score_dim = score_dim
        self.traj_embed = nn.Sequential(
            nn.Linear(poses_num * state_size, d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(d_ffn, d_model),
        )
        self.score_embed = nn.Linear(score_dim, d_model) if score_dim > 0 else None
        input_dim = d_model * (3 if score_dim > 0 else 2)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.rank_embed = nn.Embedding(max_refinement_num, d_model)
        mlp_ratio = max(float(d_ffn) / float(d_model), 1.0)
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    proj_drop=proj_drop,
                    drop_path=drop_path,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_head = MLP(d_model, d_ffn, poses_num * state_size)

    def forward(
        self,
        anchor_proposals: torch.Tensor,
        anchor_features: torch.Tensor,
        score_features: torch.Tensor,
        scene_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_anchors, _, _ = anchor_proposals.shape
        traj_features = self.traj_embed(anchor_proposals.reshape(batch_size, num_anchors, -1))
        features = [anchor_features, traj_features]
        if self.score_embed is not None:
            features.append(self.score_embed(score_features))
        tokens = self.input_proj(torch.cat(features, dim=-1))

        rank_ids = torch.arange(num_anchors, device=tokens.device).clamp_max(self.rank_embed.num_embeddings - 1)
        tokens = tokens + self.rank_embed(rank_ids)[None]
        for layer in self.layers:
            tokens = layer(tokens, scene_features)
        return self.output_head(tokens).reshape(batch_size, num_anchors, self.poses_num, self.state_size)


class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model
        self.use_external_scene_features = bool(getattr(config, "use_external_scene_features", False))
        self.external_scene_feature_skip_backbone_init = bool(
            getattr(config, "external_scene_feature_skip_backbone_init", True)
        )
        self.external_scene_feature_skip_sensors = bool(
            getattr(config, "external_scene_feature_skip_sensors", True)
        )
        self.external_scene_feature_dir = str(getattr(config, "external_scene_feature_dir", "") or "")
        self.external_scene_feature_dir_eval = str(
            getattr(config, "external_scene_feature_dir_eval", self.external_scene_feature_dir) or ""
        )
        self.external_scene_feature_file = str(getattr(config, "external_scene_feature_file", "selected_feature.pt"))
        self.external_scene_feature_dim = int(getattr(config, "external_scene_feature_dim", 3072))
        self.external_scene_feature_num_tokens = int(getattr(config, "external_scene_feature_num_tokens", 0))
        self.external_scene_feature_mode = str(
            getattr(config, "external_scene_feature_mode", "project_first")
        ).lower()
        if self.external_scene_feature_mode not in {"project_first", "kv_highdim"}:
            raise ValueError(
                f"Unsupported external_scene_feature_mode={self.external_scene_feature_mode}. "
                "Use one of: project_first, kv_highdim."
            )
        self.external_scene_feature_cache_size = max(
            0, int(getattr(config, "external_scene_feature_cache_size", 256))
        )
        self._external_scene_feature_cache: Dict[str, torch.Tensor] = {}

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        if self.use_external_scene_features and self.external_scene_feature_skip_backbone_init:
            self.num_cams = 0
            self.num_lidar = 0

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        if self.use_external_scene_features:
            if self.external_scene_feature_mode == "project_first":
                self.external_scene_proj = nn.Linear(self.external_scene_feature_dim, config.tf_d_model)
                self.external_scene_norm = nn.LayerNorm(config.tf_d_model)
            else:
                self.external_scene_norm = nn.LayerNorm(self.external_scene_feature_dim)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        scorer_cross_kv_dim = None
        if self.use_external_scene_features and self.external_scene_feature_mode == "kv_highdim":
            scorer_cross_kv_dim = self.external_scene_feature_dim

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num,
            d_model=config.tf_d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
            cross_kv_dim=scorer_cross_kv_dim,
        )

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)
        self.use_refine_scorer = bool(getattr(config, "use_refine_scorer", False))
        if self.use_refine_scorer:
            self.refine_pos_embed = copy.deepcopy(self.pos_embed)
            self.refine_scorer_attention = TransformerDecoderScorer(
                num_layers=config.scorer_ref_num,
                d_model=config.tf_d_model,
                proj_drop=0.1,
                drop_path=0.2,
                config=config,
                cross_kv_dim=scorer_cross_kv_dim,
            )
            self.refine_scorer = Scorer(config)

        self.use_refinement_head = bool(getattr(config, "use_refinement_head", False))
        if self.use_refinement_head:
            self.refinement_num = int(getattr(config, "refinement_num", 8))
            self.refinement_anchor_top_k = int(getattr(config, "refinement_anchor_top_k", self.refinement_num))
            self.refinement_delta_xy_limit = float(getattr(config, "refinement_delta_xy_limit", 2.0))
            self.refinement_delta_heading_limit = float(getattr(config, "refinement_delta_heading_limit", 0.5))
            self.refinement_detach_anchor = bool(getattr(config, "refinement_detach_anchor", True))
            self.refinement_use_score_features = bool(getattr(config, "refinement_use_score_features", True))
            self.refinement_arch = str(getattr(config, "refinement_arch", "mlp")).lower()
            refinement_score_dim = 7 if self.refinement_use_score_features else 0
            if self.refinement_arch == "decoder":
                self.refinement_head = SceneConditionedRefinementHead(
                    d_model=config.tf_d_model,
                    d_ffn=config.tf_d_ffn,
                    poses_num=self.poses_num,
                    state_size=self.state_size,
                    score_dim=refinement_score_dim,
                    num_layers=int(getattr(config, "refinement_decoder_layers", 2)),
                    num_heads=int(getattr(config, "refinement_decoder_heads", 4)),
                    max_refinement_num=max(int(getattr(config, "refinement_anchor_top_k", 8)), 16),
                    proj_drop=float(getattr(config, "refinement_decoder_proj_drop", 0.0)),
                    drop_path=float(getattr(config, "refinement_decoder_drop_path", 0.0)),
                    init_values=float(getattr(config, "refinement_decoder_ls_values", 0.0)),
                )
            elif self.refinement_arch == "mlp":
                refinement_input_dim = config.tf_d_model + refinement_score_dim
                self.refinement_head = MLP(
                    refinement_input_dim,
                    config.tf_d_ffn,
                    self.poses_num * self.state_size,
                )
            else:
                raise ValueError(f"Unsupported refinement_arch: {self.refinement_arch}")

        # region-consistency encoder (optional)
        self.use_region_consistency = bool(getattr(config, "corridor_pkl", ""))
        if self.use_region_consistency:
            self.rc_encoder = RegionConsistencyEncoder(
                d_model=config.tf_d_model,
                num_steps=config.num_poses,
            )

        self.b2d=config.b2d
        self.use_refinement_cache = bool(getattr(config, "use_refinement_cache", False))
        self.refinement_cache_path = str(getattr(config, "refinement_cache_path", ""))

    def initialize_refine_scorer_from_base(self) -> None:
        if not bool(getattr(self._config, "use_refine_scorer", False)):
            return
        self.refine_pos_embed.load_state_dict(self.pos_embed.state_dict(), strict=True)
        self.refine_scorer_attention.load_state_dict(self.scorer_attention.state_dict(), strict=True)
        self.refine_scorer.load_state_dict(self.scorer.state_dict(), strict=True)

    def _sanitize_proposals(self, proposals: torch.Tensor) -> torch.Tensor:
        """Keep proposals numerically safe, especially for closed-loop fine-tuning."""
        if not bool(getattr(self._config, "sanitize_proposals", False)):
            return proposals

        xy_limit = float(getattr(self._config, "proposal_xy_limit", 100.0))
        safe = torch.nan_to_num(proposals, nan=0.0, posinf=xy_limit, neginf=-xy_limit)
        safe_xy = safe[..., :2].clamp(min=-xy_limit, max=xy_limit)
        safe_heading = torch.atan2(torch.sin(safe[..., 2]), torch.cos(safe[..., 2]))
        return torch.cat([safe_xy, safe_heading.unsqueeze(-1)], dim=-1)

    def _load_refinement_cache(self, features: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        if not self.refinement_cache_path:
            raise RuntimeError("use_refinement_cache=true requires config.refinement_cache_path")
        tokens = features.get("scenario_token")
        if tokens is None:
            raise RuntimeError("Refinement cache requires features['scenario_token']; Dataset must provide sample tokens.")
        if isinstance(tokens, str):
            tokens = [tokens]
        cache_root = Path(self.refinement_cache_path)
        entries = []
        for token in tokens:
            token = str(token)
            path = cache_root / f"{token}.pt"
            if not path.is_file():
                if bool(getattr(self._config, "refinement_cache_fallback_to_base", True)):
                    return None
                raise FileNotFoundError(f"Missing refinement cache for token={token}: {path}")
            entries.append(torch.load(path, map_location="cpu"))

        keys = entries[0].keys()
        out = {}
        for key in keys:
            values = [entry[key] for entry in entries]
            if torch.is_tensor(values[0]):
                out[key] = torch.stack(values, dim=0).to(device=device, dtype=values[0].dtype)
        return out

    def _write_refinement_cache(
        self,
        features: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
        score_features: torch.Tensor = None,
    ) -> None:
        if not bool(getattr(self._config, "write_refinement_cache", False)):
            return
        if not self.refinement_cache_path:
            return
        tokens = features.get("scenario_token")
        if tokens is None:
            return
        if isinstance(tokens, str):
            tokens = [tokens]
        cache_root = Path(self.refinement_cache_path)
        cache_root.mkdir(parents=True, exist_ok=True)
        dtype_name = str(getattr(self._config, "refinement_cache_dtype", "float16")).lower()
        cache_dtype = torch.float16 if dtype_name == "float16" else torch.float32
        required = (
            "scene_features",
            "base_proposals",
            "base_pdm_score",
            "refinement_anchor_indices",
            "anchor_proposals",
            "anchor_features",
        )
        if any(key not in output for key in required):
            return
        for batch_index, token in enumerate(tokens):
            path = cache_root / f"{str(token)}.pt"
            if path.is_file():
                continue
            item = {
                "scene_features": output["scene_features"][batch_index].detach().cpu().to(cache_dtype),
                "base_proposals": output["base_proposals"][batch_index].detach().cpu().to(cache_dtype),
                "base_pdm_score": output["base_pdm_score"][batch_index].detach().cpu().to(torch.float32),
                "refinement_anchor_indices": output["refinement_anchor_indices"][batch_index].detach().cpu().to(torch.long),
                "anchor_proposals": output["anchor_proposals"][batch_index].detach().cpu().to(cache_dtype),
                "anchor_features": output["anchor_features"][batch_index].detach().cpu().to(cache_dtype),
            }
            if "score_features" in output:
                item["score_features"] = output["score_features"][batch_index].detach().cpu().to(cache_dtype)
            tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
            torch.save(item, tmp_path)
            try:
                tmp_path.replace(path)
            except FileExistsError:
                tmp_path.unlink(missing_ok=True)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]
        
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")


        batch_size = ego_status.shape[0]



        cached = None
        if self.use_refinement_cache:
            cached = self._load_refinement_cache(features, ego_status.device)
        cache_hit = cached is not None
        if cache_hit:
            scene_features = cached["scene_features"]
            proposals = cached["base_proposals"]
            proposal_list = [proposals]
        else:
            scene_features = self._build_scene_features(features, batch_size, ego_status.device)
            log.debug(f"Scene features - {scene_features.shape}")

            # initial trajectories
            proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
            proposals = self._sanitize_proposals(proposals)
            proposal_list = [proposals]
            log.debug(f"Proposals initial - {proposals.shape}")

            # decode the trajectories at each step of the decoder
            token_list = self.trajectory_decoder(traj_tokens, scene_features)
            log.debug(f"Trajectory decoder - {len(token_list)}")
            for i in range(self._config.ref_num):
                tokens = token_list[i]
                proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
                proposals = self._sanitize_proposals(proposals)
                proposal_list.append(proposals)
            
            traj_tokens = token_list[-1]
            proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        def _pdm_score_temperature(name: str, default: float = 1.0) -> float:
            return max(float(getattr(self._config, name, default)), 1e-6)

        t_noc = _pdm_score_temperature("pdm_score_temperature_noc")
        t_dac = _pdm_score_temperature("pdm_score_temperature_dac")
        t_ddc = _pdm_score_temperature("pdm_score_temperature_ddc")
        t_ep = _pdm_score_temperature("pdm_score_temperature_ep")
        t_ttc = _pdm_score_temperature("pdm_score_temperature_ttc")
        t_comfort = _pdm_score_temperature("pdm_score_temperature_comfort")

        def _score_candidates(candidate_proposals: torch.Tensor, scorer_branch: str = "base"):
            B, N, _, _ = candidate_proposals.shape
            detach_proposals_in_scorer = bool(getattr(self._config, "detach_proposals_in_scorer", True))
            scorer_proposals = candidate_proposals.detach() if detach_proposals_in_scorer else candidate_proposals

            if scorer_branch == "refine":
                pos_embed = self.refine_pos_embed
                scorer_attention = self.refine_scorer_attention
                scorer = self.refine_scorer
            else:
                pos_embed = self.pos_embed
                scorer_attention = self.scorer_attention
                scorer = self.scorer

            embedded_traj = pos_embed(scorer_proposals.reshape(B, N, -1))
            candidate_features = scorer_attention(embedded_traj, scene_features)
            candidate_features = candidate_features + ego_token

            if self.use_region_consistency and "corridors" in features:
                corridors = features["corridors"]
                rc_feat = self.rc_encoder(scorer_proposals, corridors)
                candidate_features = candidate_features + rc_feat

            (
                pred_logit,
                pred_logit2,
                pred_agents_states,
                pred_area_logit,
                bev_semantic_map,
                agent_states,
                agent_labels,
            ) = scorer(candidate_proposals, candidate_features)

            additive_term = (
                self._config.ttc * pred_logit["time_to_collision_within_bound"].div(t_ttc).sigmoid()
                + self._config.ep * pred_logit["ego_progress"].div(t_ep).sigmoid()
                + self._config.comfort * pred_logit["comfort"].div(t_comfort).sigmoid()
            ).clamp_min(1e-6)

            pdm_score = (
                self._config.noc * F.logsigmoid(pred_logit["no_at_fault_collisions"].div(t_noc))
                + self._config.dac * F.logsigmoid(pred_logit["drivable_area_compliance"].div(t_dac))
                + self._config.ddc * F.logsigmoid(pred_logit["driving_direction_compliance"].div(t_ddc))
                + additive_term.log()
            )
            return {
                "pred_logit": pred_logit,
                "pred_logit2": pred_logit2,
                "pred_agents_states": pred_agents_states,
                "pred_area_logit": pred_area_logit,
                "bev_semantic_map": bev_semantic_map,
                "agent_states": agent_states,
                "agent_labels": agent_labels,
                "pdm_score": pdm_score,
                "proposal_features": candidate_features,
            }

        if cache_hit:
            base_scored = {
                "pdm_score": cached["base_pdm_score"],
                "pred_logit": {},
                "proposal_features": cached["anchor_features"],
            }
        elif self.use_refinement_head:
            with torch.no_grad():
                base_scored = _score_candidates(proposals)
        else:
            base_scored = _score_candidates(proposals)
        pdm_score = base_scored["pdm_score"]
        pred_logit = base_scored.get("pred_logit", {})

        if self.use_refinement_head:
            if cache_hit:
                top_idx = cached["refinement_anchor_indices"].long()
                anchor_proposals = cached["anchor_proposals"]
                anchor_features = cached["anchor_features"]
                score_features = cached.get("score_features")
                refine_num = anchor_proposals.shape[1]
            else:
                top_k = max(1, min(int(getattr(self, "refinement_anchor_top_k", 8)), proposals.shape[1]))
                refine_num = max(1, min(int(getattr(self, "refinement_num", 8)), top_k))
                top_idx = torch.topk(pdm_score.detach(), k=top_k, dim=1).indices[:, :refine_num]
                gather_idx = top_idx[:, :, None, None].expand(-1, -1, proposals.shape[2], proposals.shape[3])
                anchor_proposals = proposals.gather(1, gather_idx)
                feature_idx = top_idx[:, :, None].expand(-1, -1, base_scored["proposal_features"].shape[-1])
                anchor_features = base_scored["proposal_features"].gather(1, feature_idx)
                score_features = None
                if self.refinement_use_score_features:
                    score_feature_list = [pdm_score.detach().unsqueeze(-1)]
                    for key in (
                        "no_at_fault_collisions",
                        "drivable_area_compliance",
                        "driving_direction_compliance",
                        "time_to_collision_within_bound",
                        "ego_progress",
                        "comfort",
                    ):
                        score_feature_list.append(base_scored["pred_logit"][key].detach().sigmoid().unsqueeze(-1))
                    score_features = torch.cat(score_feature_list, dim=-1)
                    score_features = score_features.gather(1, top_idx[:, :, None].expand(-1, -1, score_features.shape[-1]))

            if self.refinement_arch == "decoder":
                residual_raw = self.refinement_head(
                    anchor_proposals.detach(),
                    anchor_features,
                    score_features,
                    scene_features,
                )
            else:
                if score_features is not None:
                    anchor_features = torch.cat([anchor_features, score_features], dim=-1)
                residual_raw = self.refinement_head(anchor_features).reshape(
                    batch_size, refine_num, self.poses_num, self.state_size
                )
            residual_xy = torch.tanh(residual_raw[..., :2]) * self.refinement_delta_xy_limit
            residual_heading = torch.tanh(residual_raw[..., 2:3]) * self.refinement_delta_heading_limit
            residual = torch.cat([residual_xy, residual_heading], dim=-1)
            anchor_for_refine = anchor_proposals.detach() if self.refinement_detach_anchor else anchor_proposals
            refined_proposals = self._sanitize_proposals(anchor_for_refine + residual)

            mode = str(getattr(self._config, "refinement_candidate_mode", "refined_only")).lower()
            if mode == "concat":
                candidate_proposals = torch.cat([proposals, refined_proposals], dim=1)
            elif mode == "refined_only":
                candidate_proposals = refined_proposals
            else:
                raise ValueError(f"Unsupported refinement_candidate_mode: {mode}")

            final_scorer_branch = "refine" if self.use_refine_scorer else "base"
            use_mixed_concat_scorer = (
                mode == "concat"
                and self.use_refine_scorer
                and bool(getattr(self._config, "refinement_concat_mixed_scorer", False))
            )
            if use_mixed_concat_scorer:
                if not base_scored.get("pred_logit"):
                    base_scored_for_concat = _score_candidates(proposals, scorer_branch="base")
                else:
                    base_scored_for_concat = base_scored
                refined_only_scored = _score_candidates(refined_proposals, scorer_branch="refine")
                concat_pred_logit = {
                    key: torch.cat(
                        [base_scored_for_concat["pred_logit"][key], refined_only_scored["pred_logit"][key]],
                        dim=1,
                    )
                    for key in base_scored_for_concat["pred_logit"].keys()
                }
                refined_scored = {
                    "pred_logit": concat_pred_logit,
                    "pred_logit2": None,
                    "pred_agents_states": None,
                    "pred_area_logit": None,
                    "bev_semantic_map": None,
                    "agent_states": None,
                    "agent_labels": None,
                    "pdm_score": torch.cat(
                        [base_scored_for_concat["pdm_score"], refined_only_scored["pdm_score"]],
                        dim=1,
                    ),
                }
                final_scorer_branch = "mixed"
            else:
                refined_scored = _score_candidates(candidate_proposals, scorer_branch=final_scorer_branch)
            output["base_proposals"] = proposals
            output["base_pdm_score"] = pdm_score
            output["refined_proposals"] = refined_proposals
            output["refinement_anchor_indices"] = top_idx
            output["refinement_scorer_branch"] = final_scorer_branch
            if bool(getattr(self._config, "export_refinement_cache", False)) or bool(getattr(self._config, "write_refinement_cache", False)):
                output["scene_features"] = scene_features
                output["anchor_proposals"] = anchor_proposals
                output["anchor_features"] = anchor_features
                if score_features is not None:
                    output["score_features"] = score_features
            output["proposals"] = candidate_proposals
            output["proposal_list"] = proposal_list + [candidate_proposals]
            output.update({k: v for k, v in refined_scored.items() if k != "proposal_features"})
            proposals = candidate_proposals
            pred_logit = refined_scored["pred_logit"]
            pdm_score = refined_scored["pdm_score"]
            if self.use_refinement_cache and not cache_hit:
                self._write_refinement_cache(features, output)
        else:
            output.update({k: v for k, v in base_scored.items() if k != "proposal_features"})

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output

    def _select_external_feature_dir(self, training: bool) -> str:
        if training:
            return self.external_scene_feature_dir or self.external_scene_feature_dir_eval
        return self.external_scene_feature_dir_eval or self.external_scene_feature_dir

    def _load_one_external_scene_feature(self, token: str, root_dir: str) -> torch.Tensor:
        cache_key = f"{root_dir}|{token}"
        if cache_key in self._external_scene_feature_cache:
            return self._external_scene_feature_cache[cache_key]

        feature_path = Path(root_dir) / str(token) / self.external_scene_feature_file
        if not feature_path.is_file():
            raise FileNotFoundError(f"External scene feature not found: {feature_path}")

        data = torch.load(str(feature_path), map_location="cpu")
        if isinstance(data, dict):
            for key in ("selected_feature", "feature", "features", "tokens"):
                if key in data:
                    data = data[key]
                    break
            else:
                raise KeyError(
                    f"Unsupported external feature dict schema at {feature_path}, keys={list(data.keys())[:10]}"
                )
        feat = torch.as_tensor(data)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        if feat.ndim != 2:
            raise ValueError(f"Expected 2D external feature tensor at {feature_path}, got shape={tuple(feat.shape)}")
        feat = feat.to(torch.float32)

        if self.external_scene_feature_num_tokens > 0:
            t = self.external_scene_feature_num_tokens
            if feat.shape[0] > t:
                feat = feat[:t]
            elif feat.shape[0] < t:
                pad = torch.zeros((t - feat.shape[0], feat.shape[1]), dtype=feat.dtype)
                feat = torch.cat([feat, pad], dim=0)

        if self.external_scene_feature_cache_size > 0:
            if len(self._external_scene_feature_cache) >= self.external_scene_feature_cache_size:
                first_key = next(iter(self._external_scene_feature_cache))
                self._external_scene_feature_cache.pop(first_key, None)
            self._external_scene_feature_cache[cache_key] = feat
        return feat

    def _load_external_scene_features(self, features: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        if not self.use_external_scene_features:
            raise RuntimeError("_load_external_scene_features called while use_external_scene_features=False")
        tokens = features.get("scenario_token")
        if tokens is None:
            raise RuntimeError("External scene features require features['scenario_token']")
        if isinstance(tokens, str):
            tokens = [tokens]
        root_dir = self._select_external_feature_dir(training=self.training)
        if not root_dir:
            raise RuntimeError("use_external_scene_features=true but external_scene_feature_dir is empty")

        loaded: List[torch.Tensor] = [self._load_one_external_scene_feature(str(token), root_dir) for token in tokens]
        if not loaded:
            raise RuntimeError("No external scene features loaded")

        token_lens = [x.shape[0] for x in loaded]
        if len(set(token_lens)) != 1:
            common_len = min(token_lens)
            loaded = [x[:common_len] for x in loaded]
        feat = torch.stack(loaded, dim=0).to(device=device, dtype=torch.float32)
        if feat.shape[-1] != self.external_scene_feature_dim:
            raise ValueError(
                f"External feature dim mismatch: configured={self.external_scene_feature_dim}, loaded={feat.shape[-1]}"
            )
        if self.external_scene_feature_mode == "project_first":
            feat = self.external_scene_proj(feat)
            feat = self.external_scene_norm(feat)
        else:
            feat = self.external_scene_norm(feat)
        return feat

    def _build_scene_features(
        self,
        features: Dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.use_external_scene_features:
            return self._load_external_scene_features(features, device)

        scene_features = []
        # image features
        if self.num_cams > 0:
            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError("Missing image/camera_feature in features.")

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_scene_tokens = self.image_backbone(img, scene_tokens)
            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        if not scene_features:
            raise RuntimeError("No scene features available. Check sensor config or external_scene_feature_dir.")
        return torch.cat(scene_features, dim=1)
