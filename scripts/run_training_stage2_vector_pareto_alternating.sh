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
export SUBSCORE_PATH="${SUBSCORE_PATH:-${NAVSIM_EXP_ROOT}}"
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${NAVSIM_DEVKIT_ROOT}/nuplan-devkit:${PYTHONPATH:-}"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="${CLOVER_IMAGE_BACKBONE_WEIGHTS:-${REPO_ROOT}/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors}"

TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/train_metric_cache}"
EVAL_METRIC_CACHE_PATH="${EVAL_METRIC_CACHE_PATH:-${TRAIN_METRIC_CACHE_PATH}}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-}"

EXPERIMENT="${EXPERIMENT:-training_clover_stage2_vector_pareto_alternating}"
NUM_CYCLES="${NUM_CYCLES:-30}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-1}"
GENERATOR_EPOCHS="${GENERATOR_EPOCHS:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
BASE_LR="${BASE_LR:-0.00003}"
USE_RAY_FOR_SCORING="${USE_RAY_FOR_SCORING:-true}"
RAY_THREADS_PER_NODE="${RAY_THREADS_PER_NODE:-16}"
RAY_LOG_TO_DRIVER="${RAY_LOG_TO_DRIVER:-false}"
ENABLE_PROGRESS_BAR="${ENABLE_PROGRESS_BAR:-true}"
USE_TEST_LOGS_FOR_VAL="${USE_TEST_LOGS_FOR_VAL:-true}"
PARETO_GUIDANCE_WEIGHT="${PARETO_GUIDANCE_WEIGHT:-1.0}"
TEACHER_STABILITY_WEIGHT="${TEACHER_STABILITY_WEIGHT:-0.05}"
TRAJECTORY_WEIGHT="${TRAJECTORY_WEIGHT:-0.1}"
INTER_WEIGHT="${INTER_WEIGHT:-0.02}"
PARETO_SET_MAX_SIZE="${PARETO_SET_MAX_SIZE:-8}"
PARETO_MIN_SIZE="${PARETO_MIN_SIZE:-2}"
TEACHER_REWARD_THRESHOLD="${TEACHER_REWARD_THRESHOLD:-0.4}"
ALIGN_PKL_PDM_TO_CSV="${ALIGN_PKL_PDM_TO_CSV:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${OPENSCENE_DATA_ROOT}" ]; then
  echo "[Error] OPENSCENE_DATA_ROOT is not set."
  exit 1
fi
if [ -z "${NUPLAN_MAPS_ROOT}" ]; then
  echo "[Error] NUPLAN_MAPS_ROOT is not set."
  exit 1
fi
if [ -z "${INITIAL_CHECKPOINT}" ]; then
  echo "[Error] INITIAL_CHECKPOINT is not set."
  exit 1
fi
if [ ! -f "${INITIAL_CHECKPOINT}" ]; then
  echo "[Error] Initial checkpoint not found: ${INITIAL_CHECKPOINT}"
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
if [ ! -d "${EVAL_METRIC_CACHE_PATH}" ]; then
  echo "[Error] Eval metric cache directory not found: ${EVAL_METRIC_CACHE_PATH}"
  exit 1
fi

export NAVSIM_TRAIN_METRIC_CACHE_PATH="${TRAIN_METRIC_CACHE_PATH}"
export NAVSIM_EVAL_METRIC_CACHE_PATH="${EVAL_METRIC_CACHE_PATH}"
export RAY_THREADS_PER_NODE
export RAY_LOG_TO_DRIVER

if [ "${ALIGN_PKL_PDM_TO_CSV}" = "1" ]; then
  export NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1
  echo "[Train-Stage2] NAVSIM_PDM_TWO_WAY_PER_PROPOSAL=1"
else
  unset NAVSIM_PDM_TWO_WAY_PER_PROPOSAL 2>/dev/null || true
fi

MAX_EPOCHS=$((NUM_CYCLES * (CRITIC_EPOCHS + GENERATOR_EPOCHS)))

if [ "${NUM_WORKERS}" -eq 0 ]; then
  PREFETCH_FACTOR_VALUE=null
else
  PREFETCH_FACTOR_VALUE=1
fi

echo "[Train-Stage2] INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT}"
echo "[Train-Stage2] TRAIN_METRIC_CACHE_PATH=${TRAIN_METRIC_CACHE_PATH}"
echo "[Train-Stage2] EVAL_METRIC_CACHE_PATH=${EVAL_METRIC_CACHE_PATH}"
echo "[Train-Stage2] NUM_GPUS=${NUM_GPUS} NUM_WORKERS=${NUM_WORKERS} BATCH_SIZE=${BATCH_SIZE}"
echo "[Train-Stage2] NUM_CYCLES=${NUM_CYCLES} CRITIC_EPOCHS=${CRITIC_EPOCHS} GENERATOR_EPOCHS=${GENERATOR_EPOCHS}"

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training_full.py" \
  agent=clover_stage2_vector_pareto \
  experiment_name="${EXPERIMENT}" \
  train_test_split=navtrain \
  cache_path=null \
  use_cache_without_dataset=false \
  trainer.params.max_epochs="${MAX_EPOCHS}" \
  +trainer.params.devices="${NUM_GPUS}" \
  trainer.params.strategy="$([ "${NUM_GPUS}" -gt 1 ] && echo ddp_find_unused_parameters_true || echo auto)" \
  +trainer.params.log_every_n_steps=1 \
  +trainer.params.enable_progress_bar="${ENABLE_PROGRESS_BAR}" \
  trainer.params.num_sanity_val_steps=0 \
  trainer.params.default_root_dir="${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}" \
  dataloader.params.batch_size="${BATCH_SIZE}" \
  dataloader.params.num_workers="${NUM_WORKERS}" \
  dataloader.params.pin_memory=false \
  dataloader.params.prefetch_factor="${PREFETCH_FACTOR_VALUE}" \
  agent.base_checkpoint_path="${INITIAL_CHECKPOINT}" \
  agent.num_gpus="${NUM_GPUS}" \
  agent.progress_bar=false \
  agent.lr_args.name=AdamW \
  agent.lr_args.base_lr="${BASE_LR}" \
  agent.config.use_ray_for_scoring="${USE_RAY_FOR_SCORING}" \
  agent.config.alternating_stage2=true \
  agent.config.critic_phase_epochs="${CRITIC_EPOCHS}" \
  agent.config.generator_phase_epochs="${GENERATOR_EPOCHS}" \
  agent.config.stage2_training_phase=critic \
  agent.config.freeze_scorer_in_stage2=false \
  agent.config.freeze_backbone_in_stage2=true \
  agent.config.detach_proposals_in_scorer=true \
  agent.loss.stage2_training_phase=critic \
  agent.loss.pareto_guidance_weight="${PARETO_GUIDANCE_WEIGHT}" \
  agent.loss.teacher_refresh_stability_weight="${TEACHER_STABILITY_WEIGHT}" \
  agent.loss.trajectory_weight="${TRAJECTORY_WEIGHT}" \
  agent.loss.inter_weight="${INTER_WEIGHT}" \
  agent.loss.pareto_set_max_size="${PARETO_SET_MAX_SIZE}" \
  agent.loss.pareto_min_size="${PARETO_MIN_SIZE}" \
  agent.loss.teacher_reward_threshold="${TEACHER_REWARD_THRESHOLD}" \
  +auto_resume_ckpt=false \
  train_ckpt_path=null \
  +use_test_logs_for_val="${USE_TEST_LOGS_FOR_VAL}"
