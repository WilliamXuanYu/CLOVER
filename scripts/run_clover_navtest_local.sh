#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-/NEW_EDS/chency2506/navsim_workspace/dataset/maps}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/NEW_EDS/chency2506/navsim_workspace/dataset}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/outputs}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-/NEW_EDS/AsyncDriveSimulation_23/DiffusionDrive/metric_cache}"
export CHECKPOINT="${CHECKPOINT:-/NEW_EDS/recogdrive-9/score/outputs/ke/training_drivoR_stage2_vector_pareto_alternating/lightning_logs/version_11/checkpoints/21_9447.ckpt}"

DEFAULT_LOCAL_BACKBONE_IN_CLOVER="${REPO_ROOT}/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
DEFAULT_LOCAL_BACKBONE_IN_DRIVOR="/NEW_EDS/AsyncDriveSimulation_23/DrivoR-main/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
if [ -z "${CLOVER_IMAGE_BACKBONE_WEIGHTS:-}" ]; then
  if [ -f "${DEFAULT_LOCAL_BACKBONE_IN_CLOVER}" ]; then
    export CLOVER_IMAGE_BACKBONE_WEIGHTS="${DEFAULT_LOCAL_BACKBONE_IN_CLOVER}"
  elif [ -f "${DEFAULT_LOCAL_BACKBONE_IN_DRIVOR}" ]; then
    export CLOVER_IMAGE_BACKBONE_WEIGHTS="${DEFAULT_LOCAL_BACKBONE_IN_DRIVOR}"
  else
    export CLOVER_IMAGE_BACKBONE_WEIGHTS="${DEFAULT_LOCAL_BACKBONE_IN_CLOVER}"
  fi
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS="${NUM_GPUS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EXPERIMENT="${EXPERIMENT:-clover_navtest_local}"

echo "[Local Test] CHECKPOINT=${CHECKPOINT}"
echo "[Local Test] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[Local Test] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[Local Test] NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "[Local Test] METRIC_CACHE_PATH=${METRIC_CACHE_PATH}"
echo "[Local Test] CLOVER_IMAGE_BACKBONE_WEIGHTS=${CLOVER_IMAGE_BACKBONE_WEIGHTS}"

bash "${REPO_ROOT}/scripts/eval_multi_expert_navtest.sh"
