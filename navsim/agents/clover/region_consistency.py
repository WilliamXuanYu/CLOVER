"""
Region-Consistency features for DrivoR scorer.

Given predicted/GT feasible corridors [B, 8, 9] and trajectory proposals [B, N, 8, 3],
computes per-proposal geometric features that describe how well each trajectory
fits within the feasible region, then projects to scorer feature space.

Corridor format: (exists, x_bl, y_bl, x_br, y_br, x_fl, y_fl, x_fr, y_fr)
  - ego-relative coordinates, +x = forward, +y = left
  - BL/BR = back-left/right, FL/FR = front-left/right

Proposal format: (x, y, heading) per timestep, ego-relative.
"""

import torch
import torch.nn as nn


class RegionConsistencyEncoder(nn.Module):
    """
    Computes region-consistency features between proposals and corridors,
    then projects to d_model for addition to scorer input.
    """

    NUM_FEATURES_PER_STEP = 4  # exists, inside, margin_norm, alpha

    def __init__(self, d_model: int, num_steps: int = 8):
        super().__init__()
        self.num_steps = num_steps
        input_dim = num_steps * self.NUM_FEATURES_PER_STEP
        self.projector = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    @staticmethod
    def _compute_rc_features(proposals, corridors):
        """
        Pure geometric computation (differentiable w.r.t. proposals but
        we typically detach proposals before scorer anyway).

        Args:
            proposals: [B, N, T, 3] - trajectory proposals (x, y, heading)
            corridors: [B, T, 9]    - feasible corridors per timestep

        Returns:
            features: [B, N, T, 4]
        """
        B, N, T, _ = proposals.shape

        x_p = proposals[:, :, :, 0]  # [B, N, T]
        y_p = proposals[:, :, :, 1]  # [B, N, T]

        exists = corridors[:, :, 0]               # [B, T]
        x_bl = corridors[:, :, 1]; y_bl = corridors[:, :, 2]
        x_br = corridors[:, :, 3]; y_br = corridors[:, :, 4]
        x_fl = corridors[:, :, 5]; y_fl = corridors[:, :, 6]
        x_fr = corridors[:, :, 7]; y_fr = corridors[:, :, 8]

        # Longitudinal: back/front midpoints
        x_back  = (x_bl + x_br) / 2   # [B, T]
        x_front = (x_fl + x_fr) / 2   # [B, T]
        corridor_len = (x_front - x_back).clamp(min=0.1)  # [B, T]

        # alpha: longitudinal position of proposal within corridor (0=back, 1=front)
        alpha = (x_p - x_back[:, None, :]) / corridor_len[:, None, :]  # [B, N, T]
        alpha_c = alpha.clamp(0.0, 1.0)

        # Lateral boundaries interpolated at proposal's longitudinal position
        # In ego frame: y_bl > y_br (left has higher y)
        y_left  = y_bl[:, None, :] + alpha_c * (y_fl[:, None, :] - y_bl[:, None, :])
        y_right = y_br[:, None, :] + alpha_c * (y_fr[:, None, :] - y_br[:, None, :])

        # Margins (positive = inside)
        margin_l = y_left - y_p    # distance from left boundary
        margin_r = y_p - y_right   # distance from right boundary
        margin_min = torch.min(margin_l, margin_r)  # [B, N, T]

        # Inside check
        inside_lat = (margin_l > 0) & (margin_r > 0)
        inside_lon = (alpha > -0.1) & (alpha < 1.1)
        inside = (inside_lat & inside_lon).float()  # [B, N, T]

        # Apply existence mask
        ex = exists[:, None, :]  # [B, 1, T]
        inside = inside * ex
        margin_norm = (margin_min.clamp(-5, 5) / 5.0) * ex
        alpha_feat = alpha_c * ex

        features = torch.stack([
            ex.expand_as(inside),    # corridor exists
            inside,                   # point inside corridor
            margin_norm,              # normalized signed margin
            alpha_feat,               # longitudinal position
        ], dim=-1)                    # [B, N, T, 4]

        return features

    def forward(self, proposals, corridors):
        """
        Args:
            proposals: [B, N, T, 3]
            corridors: [B, T, 9]
        Returns:
            projected: [B, N, d_model]
        """
        feats = self._compute_rc_features(proposals, corridors)  # [B, N, T, 4]
        B, N, T, F = feats.shape
        flat = feats.reshape(B, N, T * F)  # [B, N, 32]
        return self.projector(flat)         # [B, N, d_model]
