# Inference Guide

Download released checkpoints from:

- `https://github.com/WilliamXuanYu/CLOVER/releases`

## Environment Variables

Before running inference, set:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/dataset/maps"
export OPENSCENE_DATA_ROOT="/PATH/TO/dataset"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/CLOVER"
export NAVSIM_EXP_ROOT="/PATH/TO/exp"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"
export METRIC_CACHE_PATH="/PATH/TO/metric_cache"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="/PATH/TO/model.safetensors"
export CHECKPOINT="/PATH/TO/clover.ckpt"
```

Optional runtime controls:

```bash
export NUM_GPUS=1
export BATCH_SIZE=64
export NUM_WORKERS=4
export EXPERIMENT="clover_navtest"
```

## Evaluation

Current public evaluation entrypoint:

```bash
bash scripts/eval_multi_expert_navtest.sh
```

Complete example:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/dataset/maps"
export OPENSCENE_DATA_ROOT="/PATH/TO/dataset"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/CLOVER"
export NAVSIM_EXP_ROOT="/PATH/TO/exp"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"
export METRIC_CACHE_PATH="/PATH/TO/metric_cache"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="/PATH/TO/model.safetensors"
export CHECKPOINT="/PATH/TO/clover.ckpt"

bash scripts/eval_multi_expert_navtest.sh
```

Outputs are written to:

- `${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}/...`
- `${SUBSCORE_PATH}/navsim1_pdm_scores/${EXPERIMENT}/...`
