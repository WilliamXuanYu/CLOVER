#!/bin/bash
# Evaluate Clover on NAVSIM-v1 navtest.
#
# This repository currently exposes inference only.
# Training code and scripts are intentionally omitted for now.
#
# Usage:
#   CHECKPOINT=/path/to/clover.ckpt \
#   OPENSCENE_DATA_ROOT=/path/to/dataset \
#   NUPLAN_MAPS_ROOT=/path/to/dataset/maps \
#   METRIC_CACHE_PATH=/path/to/metric_cache \
#   bash scripts/eval_multi_expert_navtest.sh

set -euo pipefail

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/outputs}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="${CLOVER_IMAGE_BACKBONE_WEIGHTS:-${REPO_ROOT}/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors}"

mkdir -p "${NAVSIM_EXP_ROOT}"
mkdir -p "${SUBSCORE_PATH}"

if [ -z "${CHECKPOINT:-}" ]; then
  echo "[Error] CHECKPOINT is not set."
  exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
  echo "[Error] Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi
if [ -z "${OPENSCENE_DATA_ROOT}" ]; then
  echo "[Error] OPENSCENE_DATA_ROOT is not set."
  exit 1
fi
if [ -z "${NUPLAN_MAPS_ROOT}" ]; then
  echo "[Error] NUPLAN_MAPS_ROOT is not set."
  exit 1
fi
if [ ! -f "${CLOVER_IMAGE_BACKBONE_WEIGHTS}" ]; then
  echo "[Error] Backbone weights not found: ${CLOVER_IMAGE_BACKBONE_WEIGHTS}"
  echo "[Hint] Set CLOVER_IMAGE_BACKBONE_WEIGHTS=/path/to/model.safetensors"
  exit 1
fi

METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"
if [ ! -d "${METRIC_CACHE_PATH}" ]; then
  echo "[Error] Metric cache directory not found: ${METRIC_CACHE_PATH}"
  exit 1
fi

echo "[Eval] Checkpoint: ${CHECKPOINT}"
echo "[Eval] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[Eval] NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "[Eval] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "[Eval] METRIC_CACHE_PATH=${METRIC_CACHE_PATH}"
echo "[Eval] CLOVER_IMAGE_BACKBONE_WEIGHTS=${CLOVER_IMAGE_BACKBONE_WEIGHTS}"

EXPERIMENT="${EXPERIMENT:-clover_navtest}"
AGENT="${AGENT:-clover}"
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PIN_MEMORY="${PIN_MEMORY:-false}"
SAVE_GENERATED_TRAJECTORIES="${SAVE_GENERATED_TRAJECTORIES:-false}"
SAVE_REAL_PDM_SCORES="${SAVE_REAL_PDM_SCORES:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[Eval] experiment=${EXPERIMENT}, agent=${AGENT}, NUM_GPUS=${NUM_GPUS}, BATCH_SIZE=${BATCH_SIZE}"
echo "[Eval] save_generated_trajectories=${SAVE_GENERATED_TRAJECTORIES}, save_real_pdm_scores=${SAVE_REAL_PDM_SCORES}"

ALIGN_PKL_PDM_TO_CSV="${ALIGN_PKL_PDM_TO_CSV:-0}"
if [ "${ALIGN_PKL_PDM_TO_CSV}" = "1" ]; then
  export NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1
  echo "[Eval] NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1 (PKL real_pdm aligned with CSV; expect longer predict time)"
else
  unset NAVSIM_PDM_TWO_WAY_PER_PROPOSAL 2>/dev/null || true
  echo "[Eval] NAVSIM_PDM_TWO_WAY unset (legacy 64-batch PDM in PKL)"
fi

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_multi_gpu.py" \
  train_test_split=navtest \
  "experiment_name=\"${EXPERIMENT}\"" \
  "metric_cache_path=\"${METRIC_CACHE_PATH}\"" \
  "agent=\"${AGENT}\"" \
  "agent.checkpoint_path=\"${CHECKPOINT}\"" \
  agent.config.proposal_num=64 \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.config.noc=1 \
  agent.config.dac=1 \
  agent.config.ddc=0.0 \
  agent.config.ttc=5 \
  agent.config.ep=5 \
  agent.config.comfort=2 \
  agent.config.pseudo_expert_pkl="" \
  agent.config.long_trajectory_additional_poses=2 \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.pin_memory="${PIN_MEMORY}" \
  +save_generated_trajectories="${SAVE_GENERATED_TRAJECTORIES}" \
  +save_real_pdm_scores="${SAVE_REAL_PDM_SCORES}" \
  +trainer.params.devices="${NUM_GPUS}"
