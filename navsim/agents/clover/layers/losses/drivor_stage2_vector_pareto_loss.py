from typing import Dict, List, Tuple

import numpy as np
import torch

from .drivor_loss import DrivoRLoss


class DrivoRStage2VectorParetoLoss(DrivoRLoss):
    """Alternating stage-2 loss with vector critic and Pareto guidance."""

    def __init__(
        self,
        trajectory_weight: float = 0.1,
        inter_weight: float = 0.02,
        final_score_weight: float = 1.0,
        pred_ce_weight: float = 0.0,
        pred_l1_weight: float = 0.0,
        pred_area_weight: float = 0.0,
        agent_class_weight: float = 0.0,
        agent_box_weight: float = 0.0,
        bev_semantic_weight: float = 0.0,
        pareto_guidance_weight: float = 1.0,
        pareto_set_max_size: int = 8,
        pareto_min_size: int = 2,
        teacher_reward_threshold: float = 0.4,
        teacher_refresh_stability_weight: float = 0.05,
        stage2_training_phase: str = "generator",
        **kwargs,
    ):
        super().__init__(
            trajectory_weight=trajectory_weight,
            inter_weight=inter_weight,
            sub_score_weight=0.0,
            final_score_weight=final_score_weight,
            pred_ce_weight=pred_ce_weight,
            pred_l1_weight=pred_l1_weight,
            pred_area_weight=pred_area_weight,
            prev_weight=0.0,
            agent_class_weight=agent_class_weight,
            agent_box_weight=agent_box_weight,
            bev_semantic_weight=bev_semantic_weight,
            pseudo_expert_weight=0.0,
            **kwargs,
        )
        self.pareto_guidance_weight = pareto_guidance_weight
        self.pareto_set_max_size = pareto_set_max_size
        self.pareto_min_size = pareto_min_size
        self.teacher_reward_threshold = teacher_reward_threshold
        self.teacher_refresh_stability_weight = teacher_refresh_stability_weight
        self.stage2_training_phase = stage2_training_phase

    @staticmethod
    def _stack_reward_vector(pred_logit: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ]
        return torch.stack([pred_logit[key].sigmoid() for key in keys], dim=-1)

    @staticmethod
    def _pareto_indices(values: np.ndarray) -> List[int]:
        n = values.shape[0]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            vi = values[i]
            for j in range(n):
                if i == j:
                    continue
                vj = values[j]
                if np.all(vj >= vi) and np.any(vj > vi):
                    dominated[i] = True
                    break
        return [int(i) for i in np.where(~dominated)[0]]

    def _build_teacher_pareto_set(
        self,
        teacher_proposals: torch.Tensor,
        teacher_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proposal_sets = []
        masks = []
        rewards_np = teacher_rewards.detach().cpu().numpy()
        proposals_np = teacher_proposals.detach().cpu().numpy()

        for batch_idx in range(teacher_proposals.shape[0]):
            rewards_b = rewards_np[batch_idx]
            proposals_b = proposals_np[batch_idx]
            pareto = self._pareto_indices(rewards_b)
            if len(pareto) < self.pareto_min_size:
                reward_sum = rewards_b.mean(axis=1)
                pareto = list(np.argsort(-reward_sum)[: self.pareto_min_size])

            reward_sum = rewards_b.mean(axis=1)
            pareto = [idx for idx in pareto if reward_sum[idx] >= self.teacher_reward_threshold] or pareto
            pareto = sorted(pareto, key=lambda idx: float(reward_sum[idx]), reverse=True)
            pareto = pareto[: self.pareto_set_max_size]

            mask = np.zeros(self.pareto_set_max_size, dtype=bool)
            padded = np.zeros((self.pareto_set_max_size, proposals_b.shape[1], proposals_b.shape[2]), dtype=np.float32)
            for out_idx, src_idx in enumerate(pareto):
                padded[out_idx] = proposals_b[src_idx]
                mask[out_idx] = True
            proposal_sets.append(torch.tensor(padded, dtype=torch.float32, device=teacher_proposals.device))
            masks.append(torch.tensor(mask, dtype=torch.bool, device=teacher_proposals.device))

        return torch.stack(proposal_sets, dim=0), torch.stack(masks, dim=0)

    def forward(
        self,
        targets: Dict[str, torch.Tensor],
        pred: Dict[str, torch.Tensor],
        config,
        scoring_function=None,
    ) -> Dict[str, torch.Tensor]:
        proposals = pred["proposals"]
        target_trajectory = targets["trajectory"]
        phase = str(getattr(config, "stage2_training_phase", self.stage2_training_phase)).lower()

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = scoring_function(
            targets, proposals, test=False
        )

        min_loss = torch.linalg.norm(
            proposals - target_trajectory[:, None], dim=-1, ord=1
        ).mean(-1).amin(1).mean()
        inter_loss = self.diversity_loss(proposals)
        trajectory_loss = min_loss + self.inter_weight * inter_loss

        l2_distance = -((proposals.detach() - target_trajectory[:, None]) ** 2) / 0.5
        sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
            pred["pred_logit"],
            pred["pred_logit2"],
            pred["pred_agents_states"],
            pred["pred_area_logit"],
            target_scores,
            gt_states,
            gt_valid,
            gt_ego_areas,
            l2_distance.detach(),
        )

        pred_reward_vector = self._stack_reward_vector(pred["pred_logit"])
        real_reward_vector = target_scores[:, :, :6]

        real_reward_mean = real_reward_vector.mean(dim=-1)
        pred_reward_mean = pred_reward_vector.mean(dim=-1)
        reward_alignment = torch.abs(pred_reward_mean.detach() - pred_reward_mean).mean() * 0.0

        pareto_guidance_loss = proposals.new_tensor(0.0)
        pareto_set_size = proposals.new_tensor(0.0)
        teacher_stability_loss = proposals.new_tensor(0.0)

        if phase == "generator" and "teacher_proposals" in pred and "teacher_pred_logit" in pred:
            teacher_proposals = pred["teacher_proposals"]
            teacher_reward_vector = self._stack_reward_vector(pred["teacher_pred_logit"])
            teacher_pareto, teacher_mask = self._build_teacher_pareto_set(teacher_proposals, teacher_reward_vector)
            pareto_guidance_loss = self.multi_expert_coverage_loss(proposals, teacher_pareto, teacher_mask)
            pareto_set_size = teacher_mask.float().sum(dim=1).mean()
            teacher_stability_loss = torch.linalg.norm(
                proposals - teacher_proposals.detach(), dim=-1, ord=1
            ).mean(-1).amin(1).mean()

        loss = final_score_loss
        if phase == "critic":
            loss = (
                self.final_score_weight * final_score_loss
                + self.pred_ce_weight * pred_ce_loss
                + self.pred_l1_weight * pred_l1_loss
                + self.pred_area_weight * pred_area_loss
            )
        elif phase == "generator":
            loss = (
                self.trajectory_weight * trajectory_loss
                + self.pareto_guidance_weight * pareto_guidance_loss
                + self.teacher_refresh_stability_weight * teacher_stability_loss
            )

        top_proposals = torch.argmax(pred_reward_mean.detach(), dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()
        real_pareto_mean = real_reward_mean.topk(min(self.pareto_min_size, real_reward_mean.shape[1]), dim=1).values.mean()
        pred_pareto_mean = pred_reward_mean.topk(min(self.pareto_min_size, pred_reward_mean.shape[1]), dim=1).values.mean()

        [da_loss, ttc_loss, noc_loss, progress_loss, ddc_loss, comfort_loss] = sub_score_loss

        return {
            "loss": loss,
            "phase": proposals.new_tensor(0.0 if phase == "critic" else 1.0),
            "trajectory_loss": trajectory_loss,
            "final_score_loss": final_score_loss,
            "pareto_guidance_loss": pareto_guidance_loss,
            "teacher_stability_loss": teacher_stability_loss,
            "real_pareto_mean": real_pareto_mean,
            "pred_pareto_mean": pred_pareto_mean,
            "pareto_set_size": pareto_set_size,
            "da_loss": da_loss,
            "ttc_loss": ttc_loss,
            "noc_loss": noc_loss,
            "progress_loss": progress_loss,
            "ddc_loss": ddc_loss,
            "comfort_loss": comfort_loss,
            "inter_loss": inter_loss,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score,
            "reward_alignment": reward_alignment,
        }
