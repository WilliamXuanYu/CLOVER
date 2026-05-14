"""
Pseudo-expert trajectory loader and selector for multi-expert training.

Loads decoupled_v2 PKL data, selects diverse high-score trajectories
as pseudo-experts to provide multi-modal supervision for the generator.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_pseudo_expert_index(pkl_path: str) -> Dict[str, Dict]:
    """
    Load decoupled_v2 PKL and build a token -> scene_data index.
    Only keeps valid scenes with at least one trajectory.
    """
    pkl_path = str(pkl_path)
    if not pkl_path or not Path(pkl_path).exists():
        return {}

    logger.info(f"Loading pseudo-expert PKL: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    index: Dict[str, Dict] = {}
    for scene in data:
        if not scene.get("valid", False):
            continue
        token = scene["token"]
        trajs = scene.get("trajectories_relative", [])
        scores = scene.get("scores", [])
        if len(trajs) == 0 or len(scores) == 0:
            continue
        index[token] = {
            "trajectories": [np.array(t, dtype=np.float32) for t in trajs],
            "scores": scores,
        }

    logger.info(f"Pseudo-expert index: {len(index)} scenes loaded")
    return index


def select_pseudo_experts(
    scene_data: Dict,
    original_expert: np.ndarray,
    top_k: int = 8,
    score_thr: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select diverse pseudo-expert trajectories for a scene.

    Strategy:
    1. Filter by pdm_score >= threshold
    2. Greedy farthest-point sampling for diversity
    3. Always include original expert if not already covered

    Returns:
        experts: np.ndarray [top_k, num_poses, 3] padded
        mask:    np.ndarray [top_k] boolean, True for valid entries
    """
    trajs = scene_data["trajectories"]
    scores = scene_data["scores"]

    candidates: List[Tuple[np.ndarray, float]] = []
    for traj, sc in zip(trajs, scores):
        pdm = sc["pdm_score"]
        if pdm >= score_thr:
            candidates.append((np.array(traj, dtype=np.float32), pdm))

    num_poses = original_expert.shape[0]

    if len(candidates) == 0:
        experts = np.zeros((top_k, num_poses, 3), dtype=np.float32)
        mask = np.zeros(top_k, dtype=bool)
        experts[0] = original_expert
        mask[0] = True
        return experts, mask

    candidates.sort(key=lambda x: -x[1])

    # Greedy farthest-point sampling
    selected: List[np.ndarray] = [candidates[0][0]]
    used = {0}

    for _ in range(min(top_k - 1, len(candidates) - 1)):
        best_idx = -1
        best_min_dist = -1.0
        for i, (cand, _) in enumerate(candidates):
            if i in used:
                continue
            min_dist = min(
                np.abs(cand[:, :2] - s[:, :2]).mean() for s in selected
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx < 0 or best_min_dist < 0.05:
            break
        selected.append(candidates[best_idx][0])
        used.add(best_idx)

    # Include original expert if not covered
    expert_covered = any(
        np.abs(original_expert[:, :2] - s[:, :2]).mean() < 0.5
        for s in selected
    )
    if not expert_covered and len(selected) < top_k:
        selected.append(original_expert.astype(np.float32))

    # Pad to top_k
    experts = np.zeros((top_k, num_poses, 3), dtype=np.float32)
    mask = np.zeros(top_k, dtype=bool)
    for i, s in enumerate(selected):
        experts[i] = s
        mask[i] = True

    return experts, mask
