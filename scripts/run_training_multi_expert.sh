#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HYDRA_FULL_ERROR=1
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${REPO_ROOT}}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${REPO_ROOT}/outputs}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="${CLOVER_IMAGE_BACKBONE_WEIGHTS:-${REPO_ROOT}/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors}"

EXPERIMENT="${EXPERIMENT:-training_clover_nav1_multi_expert}"
RUN_UID="${RUN_UID:-$(date +%m.%d_%H.%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}/${RUN_UID}}"

TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/train_metric_cache}"
PSEUDO_EXPERT_PKL="${PSEUDO_EXPERT_PKL:-}"
PE_TOP_K="${PE_TOP_K:-8}"
PE_SCORE_THR="${PE_SCORE_THR:-0.8}"
PE_WEIGHT="${PE_WEIGHT:-0.5}"
PE_LOSS_MODE="${PE_LOSS_MODE:-final_only}"

NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-false}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
BASE_LR="${BASE_LR:-0.0002}"
DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-1}"
USE_RAY_FOR_SCORING="${USE_RAY_FOR_SCORING:-true}"
RAY_THREADS_PER_NODE="${RAY_THREADS_PER_NODE:-16}"
AUTO_RESUME_TRAINING="${AUTO_RESUME_TRAINING:-false}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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
  exit 1
fi
if [ ! -d "${TRAIN_METRIC_CACHE_PATH}" ]; then
  echo "[Error] Train metric cache directory not found: ${TRAIN_METRIC_CACHE_PATH}"
  exit 1
fi

case "${PE_LOSS_MODE}" in
  final_only)
    PE_ALL_REFINEMENT=false
    ;;
  all_refinement)
    PE_ALL_REFINEMENT=true
    ;;
  *)
    echo "[Error] Unsupported PE_LOSS_MODE=${PE_LOSS_MODE}. Use final_only or all_refinement."
    exit 1
    ;;
esac

if [ "${NUM_WORKERS}" -eq 0 ]; then
  PREFETCH_FACTOR_VALUE=null
else
  PREFETCH_FACTOR_VALUE="${DATALOADER_PREFETCH_FACTOR}"
fi

export NAVSIM_TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH}"
export RAY_THREADS_PER_NODE

mkdir -p "${OUTPUT_DIR}"

echo "[Train-Stage1] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[Train-Stage1] TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH}"
echo "[Train-Stage1] PSEUDO_EXPERT_PKL=${PSEUDO_EXPERT_PKL:-<disabled>}"
echo "[Train-Stage1] NUM_GPUS=${NUM_GPUS} NUM_WORKERS=${NUM_WORKERS} BATCH_SIZE=${BATCH_SIZE}"

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py" \
  experiment_uid="${RUN_UID}" \
  output_dir="${OUTPUT_DIR}" \
  agent=clover \
  experiment_name="${EXPERIMENT}" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.prefetch_factor="${PREFETCH_FACTOR_VALUE}" \
  dataloader.params.pin_memory="${PIN_MEMORY}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="${BASE_LR}" \
  agent.num_gpus="${NUM_GPUS}" \
  agent.progress_bar=false \
  agent.config.refiner_ls_values=0.0 \
  agent.config.image_backbone.focus_front_cam=false \
  agent.config.one_token_per_traj=true \
  agent.config.refiner_num_heads=1 \
  agent.config.tf_d_model=256 \
  agent.config.tf_d_ffn=1024 \
  agent.config.area_pred=false \
  agent.config.agent_pred=false \
  agent.config.ref_num=4 \
  agent.loss.prev_weight=0.0 \
  agent.config.long_trajectory_additional_poses=2 \
  agent.config.pseudo_expert_pkl="${PSEUDO_EXPERT_PKL}" \
  +agent.config.pseudo_expert_top_k="${PE_TOP_K}" \
  +agent.config.pseudo_expert_score_thr="${PE_SCORE_THR}" \
  +agent.loss.pseudo_expert_weight="${PE_WEIGHT}" \
  +agent.loss.pseudo_expert_all_refinement="${PE_ALL_REFINEMENT}" \
  trainer.params.strategy="$([ "${NUM_GPUS}" -gt 1 ] && echo ddp || echo auto)" \
  +trainer.params.devices="${NUM_GPUS}" \
  trainer.params.num_sanity_val_steps=0 \
  +trainer.params.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
  distributed_timeout_seconds="${DDP_TIMEOUT}" \
  +agent.config.use_ray_for_scoring="${USE_RAY_FOR_SCORING}" \
  +auto_resume_ckpt="${AUTO_RESUME_TRAINING}" \
  seed=2
