# Training Guide

This document describes the released CLOVER training pipeline on NAVSIM-v1:

1. Cache train metric data for PDM score computation.
2. Run stage-1 multi-expert training.
3. Run stage-2 vector-Pareto alternating training.

## Prerequisites

Set the shared environment variables first:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/dataset/maps"
export OPENSCENE_DATA_ROOT="/PATH/TO/dataset"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/CLOVER"
export NAVSIM_EXP_ROOT="/PATH/TO/exp"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="/PATH/TO/model.safetensors"
```

Optional overrides:

```bash
export TRAIN_METRIC_CACHE_PATH="${NAVSIM_EXP_ROOT}/train_metric_cache"
export EVAL_METRIC_CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache"
export NUM_GPUS=4
export NUM_WORKERS=8
export BATCH_SIZE=32
```

For stage-1 multi-expert training, also prepare:

```bash
export PSEUDO_EXPERT_PKL="/PATH/TO/pseudo_expert.pkl"
```

The stage-1 multi-expert pipeline depends on evaluator-filtered pseudo-expert trajectories. The released pseudo-expert package is distributed via:

- `https://drive.google.com/drive/folders/1oNTv5Pe-naw_i81rqaKk8KIs0VcUqGZ-?usp=drive_link`

## Step 1: Cache Train Metrics

The train metric cache is required before training because CLOVER computes real PDM-aligned supervision during optimization.

Direct command:

```bash
python navsim/planning/script/run_train_metric_caching.py
```

Convenience wrapper:

```bash
bash scripts/run_train_metric_caching.sh
```

By default the cache is written to:

```bash
${NAVSIM_EXP_ROOT}/train_metric_cache
```

## Step 2: Stage-1 Multi-Expert Training

Run:

```bash
bash scripts/run_training_multi_expert.sh
```

Useful overrides:

```bash
NUM_GPUS=4 \
NUM_WORKERS=8 \
BATCH_SIZE=32 \
PSEUDO_EXPERT_PKL=/PATH/TO/pseudo_expert.pkl \
bash scripts/run_training_multi_expert.sh
```

Pseudo-expert controls:

```bash
export PE_TOP_K=8
export PE_SCORE_THR=0.8
export PE_WEIGHT=0.5
export PE_LOSS_MODE=final_only
```

If `PSEUDO_EXPERT_PKL` is empty, the script falls back to vanilla single-trajectory training.

Stage-1 checkpoints are written under:

```bash
${NAVSIM_EXP_ROOT}/ke/training_clover_nav1_multi_expert/...
```

## Step 3: Stage-2 Vector-Pareto Alternating Training

Set the stage-1 checkpoint:

```bash
export INITIAL_CHECKPOINT="/PATH/TO/stage1.ckpt"
```

Run:

```bash
bash scripts/run_training_stage2_vector_pareto_alternating.sh
```

Useful overrides:

```bash
INITIAL_CHECKPOINT=/PATH/TO/stage1.ckpt \
TRAIN_METRIC_CACHE_PATH="${NAVSIM_EXP_ROOT}/train_metric_cache" \
EVAL_METRIC_CACHE_PATH="${NAVSIM_EXP_ROOT}/metric_cache" \
NUM_GPUS=4 \
NUM_WORKERS=8 \
BATCH_SIZE=32 \
bash scripts/run_training_stage2_vector_pareto_alternating.sh
```

Stage-2 controls:

```bash
export NUM_CYCLES=30
export CRITIC_EPOCHS=1
export GENERATOR_EPOCHS=1
export USE_TEST_LOGS_FOR_VAL=true
export USE_RAY_FOR_SCORING=true
```

If you want the per-proposal real PDM path aligned with the CSV evaluation path, enable:

```bash
export ALIGN_PKL_PDM_TO_CSV=1
```

This is slower and is disabled by default.

Stage-2 checkpoints are written under:

```bash
${NAVSIM_EXP_ROOT}/ke/training_clover_stage2_vector_pareto_alternating/...
```

## Notes

- `scripts/run_training_multi_expert.sh` and `scripts/run_training_stage2_vector_pareto_alternating.sh` are intentionally minimal public launchers. They do not include cluster-specific keepalive, watchdog, or scheduler workarounds.
- The released training path only includes the main stage-1 and stage-2 pipeline. Other internal ablations and side experiments are not part of this repository.
