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

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_WORKERS="${MAX_WORKERS:-${NUM_WORKERS}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "${OPENSCENE_DATA_ROOT}" ]; then
  echo "[Error] OPENSCENE_DATA_ROOT is not set."
  exit 1
fi
if [ -z "${NUPLAN_MAPS_ROOT}" ]; then
  echo "[Error] NUPLAN_MAPS_ROOT is not set."
  exit 1
fi

echo "[Cache] NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT}"
echo "[Cache] NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "[Cache] OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "[Cache] NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "[Cache] NUM_WORKERS=${NUM_WORKERS}"

"${PYTHON_BIN}" "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_train_metric_caching.py" \
  worker=ray_distributed_no_torch \
  max_number_of_workers="${MAX_WORKERS}" \
  worker.threads_per_node="${NUM_WORKERS}"
