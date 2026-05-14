import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Any, Dict, Tuple, List
from navsim.common.dataclasses import Trajectory
from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, for_viz = False):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None
        self.for_viz = for_viz

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples.
        Only the scorer-selected trajectory is evaluated via real PDM scoring,
        matching the eval pipeline (1 prediction + expert 2-way) and avoiding
        the 64-proposal batched EP normalisation mismatch.
        """
        if 'drivor' in self.agent.name() or "DrivoR" in self.agent.name():
            features, targets = batch
            batch_size = int(features["ego_status"].shape[0])
            predictions = self.agent.forward(features)

            selected_traj = predictions["trajectory"].unsqueeze(1)  # [B, 1, T, 3]
            _, _, top1_scores, _, top1_metrics = self.agent.compute_score(
                targets, selected_traj
            )
            # top1_scores: [B, 1], top1_metrics: [B, 1, 7]
            top1_score = top1_scores[:, 0].mean()
            trajectory_scores = top1_metrics[:, 0]  # [B, 7]

            target_trajectory = targets["trajectory"]
            l2 = torch.linalg.norm(
                predictions["trajectory"] - target_trajectory, dim=-1
            )[:, :4].mean()

            logging_prefix = "val"

            if "pdm_score" in predictions:
                pred_best = predictions["pdm_score"].max(dim=1).values
                score_error = torch.abs(pred_best - top1_scores[:, 0]).mean()
                self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

            self.log(f"{logging_prefix}/top1_score", top1_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/score", top1_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/collision", trajectory_scores[:, 0].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/dac", trajectory_scores[:, 1].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/progress", trajectory_scores[:, 2].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/ttc", trajectory_scores[:, 3].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{logging_prefix}/comfort", trajectory_scores[:, 4].mean(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

            return top1_score
        else:
            return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
    
    def predict_step(self, batch: Any, batch_idx: int):
        """
        Used during the multi-gpu proccessing to parallelize the prediction of trajectories.
        NOTE: requires append_token_to_batch=True in the dataset used to instantiate the trainer.
        """
        return self.predict_step_drivor(batch, batch_idx)

    def predict_step_drivor(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor], List[str]], batch_idx: int):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions = self.agent.forward(features)
            poses = predictions["trajectory"]
        result = {}
        if self.for_viz:
            final_trajectories = predictions["proposals"]
            _, _, final_scores, _, full_metrics = self.agent.compute_score(targets, final_trajectories)
            scorer_scores = predictions["pdm_score"]
            pred_logit = predictions["pred_logit"]
            ego_status = features["ego_status"]

            for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
                proposal = Trajectory(pose)
                # Keep legacy pkl format:
                # - all_proposals: final 64 trajectories only, shape [64, 8, 3]
                # - scorer_scores: model scorer output for 64 trajectories
                # - real_pdm_scores: NAVSIM real PDM scores for 64 trajectories
                # - scorer_subscores: model scorer subscore predictions for 64 trajectories
                # - real_pdm_subscores: NAVSIM real PDM subscore values for 64 trajectories
                all_proposals = final_trajectories[index].detach().cpu().numpy()
                result[token] = {
                    'trajectory': proposal,
                    'all_proposals': all_proposals,
                    'scorer_scores': scorer_scores[index].detach().cpu().numpy(),
                    'real_pdm_scores': final_scores[index].detach().cpu().numpy(),
                    'scorer_subscores': {
                        'no_at_fault_collisions': pred_logit['no_at_fault_collisions'][index].sigmoid().detach().cpu().numpy(),
                        'drivable_area_compliance': pred_logit['drivable_area_compliance'][index].sigmoid().detach().cpu().numpy(),
                        'driving_direction_compliance': pred_logit['driving_direction_compliance'][index].sigmoid().detach().cpu().numpy(),
                        'time_to_collision_within_bound': pred_logit['time_to_collision_within_bound'][index].sigmoid().detach().cpu().numpy(),
                        'ego_progress': pred_logit['ego_progress'][index].sigmoid().detach().cpu().numpy(),
                        'comfort': pred_logit['comfort'][index].sigmoid().detach().cpu().numpy(),
                    },
                    'real_pdm_subscores': {
                        'no_at_fault_collisions': full_metrics[index, :, 0].detach().cpu().numpy(),
                        'drivable_area_compliance': full_metrics[index, :, 1].detach().cpu().numpy(),
                        'ego_progress': full_metrics[index, :, 2].detach().cpu().numpy(),
                        'time_to_collision_within_bound': full_metrics[index, :, 3].detach().cpu().numpy(),
                        'comfort': full_metrics[index, :, 4].detach().cpu().numpy(),
                        'driving_direction_compliance': full_metrics[index, :, 5].detach().cpu().numpy(),
                    },
                    # Backward-compat alias used in some internal scripts.
                    'all_proposal_scores': final_scores[index].detach().cpu().numpy(),
                    'high_level_command': ego_status[index]
                }
            return result

        for index, (pose, token) in enumerate(zip(poses.cpu().numpy(), tokens)):
            proposal = Trajectory(pose)
            result[token] = {'trajectory': proposal}
        return result
