"""
PDM 打分（供 DrivoR compute_score / 训练与 eval 使用）。

默认：64 条候选一次 batch 仿真与打分。当 metric cache 含 pdm_progress（训练 cache）
或 trajectory（标准 navtest cache）时，EP 按逐条 vs 专家归一化，与 2-way 对齐；
否则退回全局 max 归一化。

可选：设置环境变量 NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1 时，对每条候选单独做
[专家轨迹, 该候选] 两轨迹打分，与 navsim/evaluate/pdm_score.pdm_score 及 CSV 口径一致（较慢）。
"""

from __future__ import annotations

import lzma
import os
import pickle
from typing import Any, List

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex

from .train_pdm_scorer import PDMScorer, PDMScorerConfig

proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig()
scorer = PDMScorer(proposal_sampling, config)

_PRED_IDX = 1

# 评测 / 导出 PKL 与 CSV 对齐时开启（见 scripts/eval_multi_expert_navtest.sh）
_ENV_TWO_WAY = "NAVSIM_PDM_TWO_WAY_PER_PROPOSAL"


def _weighted_metric_member(*names):
    for name in names:
        member = getattr(WeightedMetricIndex, name, None)
        if member is not None:
            return member
    return None


def _multi_metric_member(*names):
    for name in names:
        member = getattr(MultiMetricIndex, name, None)
        if member is not None:
            return member
    return None


_WEIGHTED_PROGRESS = _weighted_metric_member("PROGRESS")
_WEIGHTED_TTC = _weighted_metric_member("TTC")
_WEIGHTED_COMFORT = _weighted_metric_member("COMFORTABLE", "HISTORY_COMFORT")
_WEIGHTED_DRIVING_DIRECTION = _weighted_metric_member("DRIVING_DIRECTION")
_MULTI_DRIVING_DIRECTION = _multi_metric_member("DRIVING_DIRECTION")


def _normalize_metric_cache_path_compat(path: str) -> str:
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


def _two_way_per_proposal_enabled() -> bool:
    v = os.environ.get(_ENV_TWO_WAY, "")
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _extract_comfort_scores() -> np.ndarray:
    if _WEIGHTED_COMFORT is not None:
        return scorer._weighted_metrics[_WEIGHTED_COMFORT, :]
    return np.ones(scorer._weighted_metrics.shape[1], dtype=np.float64)


def _extract_driving_direction_scores() -> np.ndarray:
    if _WEIGHTED_DRIVING_DIRECTION is not None:
        return scorer._weighted_metrics[_WEIGHTED_DRIVING_DIRECTION, :]
    if _MULTI_DRIVING_DIRECTION is not None:
        return scorer._multi_metrics[_MULTI_DRIVING_DIRECTION, :]
    return np.ones(scorer._weighted_metrics.shape[1], dtype=np.float64)


def _extract_comfort_score_at(index: int) -> float:
    if _WEIGHTED_COMFORT is not None:
        return float(scorer._weighted_metrics[_WEIGHTED_COMFORT, index])
    return 1.0


def _extract_driving_direction_score_at(index: int) -> float:
    if _WEIGHTED_DRIVING_DIRECTION is not None:
        return float(scorer._weighted_metrics[_WEIGHTED_DRIVING_DIRECTION, index])
    if _MULTI_DRIVING_DIRECTION is not None:
        return float(scorer._multi_metrics[_MULTI_DRIVING_DIRECTION, index])
    return 1.0


def get_scores(args: List[Any]) -> List[Any]:
    return [
        get_sub_score(
            a["token"],
            a["poses"],
            a["test"],
            a.get("return_full_metrics", True),
        )
        for a in args
    ]


def get_sub_score(metric_cache, poses, test, return_full_metrics=True):
    # Ray / 多进程下须再次规范化：worker 可能未继承 SUBST 环境变量，但路径仍来自旧 metadata。
    metric_cache = str(metric_cache).replace("JJ_Group/", "", 1)
    metric_cache = _normalize_metric_cache_path_compat(str(metric_cache))
    with lzma.open(metric_cache, "rb") as f:
        mc = pickle.load(f)

    if _two_way_per_proposal_enabled():
        return _get_sub_score_two_way(mc, poses, test, return_full_metrics)
    return _get_sub_score_batched(mc, poses, test, return_full_metrics)


def _compute_expert_pdm_progress(mc, initial_ego_state):
    """Compute expert trajectory's weighted progress for per-proposal EP normalization.

    When the metric cache has mc.trajectory (standard navtest cache) but not
    mc.pdm_progress (train cache), we simulate the expert once to derive the
    same reference value that the train cache stores.  This makes the batched
    EP normalisation identical to 2-way scoring without running 64x simulations.
    """
    pdm_states = get_trajectory_as_array(
        mc.trajectory, proposal_sampling, initial_ego_state.time_point
    )
    exp_simulated = simulator.simulate_proposals(pdm_states[None], initial_ego_state)
    exp_scorer = PDMScorer(proposal_sampling, PDMScorerConfig())
    exp_scorer.score_proposals(
        exp_simulated,
        mc.observation,
        mc.centerline,
        mc.route_lane_ids,
        mc.drivable_area_map,
    )
    exp_multi = exp_scorer._multi_metrics.prod(axis=0)
    return float(exp_scorer._progress_raw[0] * exp_multi[0])


def _get_sub_score_batched(mc, poses, test, return_full_metrics):
    """默认：仅 64 条预测轨迹一批（原行为）。"""
    initial_ego_state = mc.ego_state

    trajectory_states = []
    for model_trajectory in poses:
        pred_trajectory = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_trajectory, simulator.proposal_sampling, initial_ego_state.time_point
        )
        trajectory_states.append(pred_states)

    trajectory_states = np.stack(trajectory_states, axis=0)
    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    pdm_progress = getattr(mc, "pdm_progress", None)
    if pdm_progress is None and hasattr(mc, "trajectory") and mc.trajectory is not None:
        pdm_progress = _compute_expert_pdm_progress(mc, initial_ego_state)

    final_scores = scorer.score_proposals(
        simulated_states,
        mc.observation,
        mc.centerline,
        mc.route_lane_ids,
        mc.drivable_area_map,
        pdm_progress,
    )

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    driving_direction_compliance = _extract_driving_direction_scores()
    ego_progress = scorer._weighted_metrics[_WEIGHTED_PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[_WEIGHTED_TTC, :]
    comfort = _extract_comfort_scores()

    if test and not return_full_metrics:
        return final_scores.astype(np.float32, copy=False)

    scores = np.stack(
        [
            no_at_fault_collisions,
            drivable_area_compliance,
            ego_progress,
            time_to_collision_within_bound,
            comfort,
            driving_direction_compliance,
            final_scores,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    if test:
        return scores

    num_col = 2
    num_p = len(final_scores)
    key_agent_corners = np.zeros(
        [num_p, scorer.proposal_sampling.num_poses, num_col, 4, 2],
        dtype=np.float32,
    )
    key_agent_labels = np.zeros(
        [num_p, scorer.proposal_sampling.num_poses, num_col],
        dtype=bool,
    )
    ego_areas = scorer._ego_areas[:, 1:, 1:]

    for i in range(len(scores)):
        proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
        if len(proposal_fault_collided_track_ids):
            col_token = proposal_fault_collided_track_ids[0]
            collision_time_idcs = int(scorer._collision_time_idcs[i]) + 1

            for time_idx in range(1, collision_time_idcs):
                if col_token in scorer._observation[time_idx].tokens:
                    key_agent_labels[i][time_idx - 1, 0] = True
                    key_agent_corners[i][time_idx - 1, 0] = np.array(
                        scorer._observation[time_idx][col_token].boundary.xy
                    ).T[:4]

        ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]
        if len(ttc_collided_track_ids):
            ttc_token = ttc_collided_track_ids[0]
            ttc_time_idcs = int(scorer._ttc_time_idcs[i]) + 1

            for time_idx in range(1, ttc_time_idcs):
                if ttc_token in scorer._observation[time_idx].tokens:
                    key_agent_labels[i][time_idx - 1, 1] = True
                    key_agent_corners[i][time_idx - 1, 1] = np.array(
                        scorer._observation[time_idx][ttc_token].boundary.xy
                    ).T[:4]

    theta = initial_ego_state.rear_axle.heading
    origin_x = initial_ego_state.rear_axle.x
    origin_y = initial_ego_state.rear_axle.y
    c, s = np.cos(theta), np.sin(theta)
    mat = np.array([[c, -s], [s, c]])
    key_agent_corners[..., 0] -= origin_x
    key_agent_corners[..., 1] -= origin_y
    key_agent_corners = key_agent_corners.dot(mat)

    return scores, key_agent_corners, key_agent_labels, ego_areas


def _get_sub_score_two_way(mc, poses, test, return_full_metrics):
    """NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1：每条候选与专家轨迹 2-way，对齐 CSV。"""
    initial_ego_state = mc.ego_state
    pdm_states = get_trajectory_as_array(
        mc.trajectory,
        proposal_sampling,
        initial_ego_state.time_point,
    )

    num_p = len(poses)
    metric_rows: List[np.ndarray] = []

    num_col = 2
    key_agent_corners = np.zeros(
        [num_p, scorer.proposal_sampling.num_poses, num_col, 4, 2],
        dtype=np.float32,
    )
    key_agent_labels = np.zeros(
        [num_p, scorer.proposal_sampling.num_poses, num_col],
        dtype=bool,
    )
    ego_area_slices: List[np.ndarray] = []

    for i, model_trajectory in enumerate(poses):
        pred_traj = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)
        pred_states = get_trajectory_as_array(
            pred_traj,
            proposal_sampling,
            initial_ego_state.time_point,
        )
        two = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)
        simulated = simulator.simulate_proposals(two, initial_ego_state)
        scores_vec = scorer.score_proposals(
            simulated,
            mc.observation,
            mc.centerline,
            mc.route_lane_ids,
            mc.drivable_area_map,
        )

        metric_rows.append(
            np.array(
                [
                    float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, _PRED_IDX]),
                    float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, _PRED_IDX]),
                    float(scorer._weighted_metrics[_WEIGHTED_PROGRESS, _PRED_IDX]),
                    float(scorer._weighted_metrics[_WEIGHTED_TTC, _PRED_IDX]),
                    _extract_comfort_score_at(_PRED_IDX),
                    _extract_driving_direction_score_at(_PRED_IDX),
                    float(scores_vec[_PRED_IDX]),
                ],
                dtype=np.float32,
            )
        )

        if not test:
            ego_area_slices.append(np.array(scorer._ego_areas[_PRED_IDX, 1:, 1:], copy=True))

            fault_ids = scorer.proposal_fault_collided_track_ids[_PRED_IDX]
            if len(fault_ids):
                col_token = fault_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[_PRED_IDX]) + 1

                for time_idx in range(1, collision_time_idcs):
                    if col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx - 1, 0] = True
                        key_agent_corners[i][time_idx - 1, 0] = np.array(
                            scorer._observation[time_idx][col_token].boundary.xy
                        ).T[:4]

            ttc_ids = scorer.ttc_collided_track_ids[_PRED_IDX]
            if len(ttc_ids):
                ttc_token = ttc_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[_PRED_IDX]) + 1

                for time_idx in range(1, ttc_time_idcs):
                    if ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx - 1, 1] = True
                        key_agent_corners[i][time_idx - 1, 1] = np.array(
                            scorer._observation[time_idx][ttc_token].boundary.xy
                        ).T[:4]

    scores = np.stack(metric_rows, axis=0)

    if test and not return_full_metrics:
        return scores[:, -1].astype(np.float32, copy=False)

    if test:
        return scores

    theta = initial_ego_state.rear_axle.heading
    origin_x = initial_ego_state.rear_axle.x
    origin_y = initial_ego_state.rear_axle.y
    c, s = np.cos(theta), np.sin(theta)
    mat = np.array([[c, -s], [s, c]])
    key_agent_corners[..., 0] -= origin_x
    key_agent_corners[..., 1] -= origin_y
    key_agent_corners = key_agent_corners.dot(mat)

    ego_areas = np.stack(ego_area_slices, axis=0)
    return scores, key_agent_corners, key_agent_labels, ego_areas
