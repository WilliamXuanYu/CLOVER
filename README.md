# CLOVER

<p align="center">
  <img src="fig/image1.png" alt="Pipeline">
</p>

End-to-end autonomous driving planners are commonly trained by imitating a single logged trajectory, yet they are evaluated by rule-based planning metrics that measure safety, feasibility, progress, and comfort. This creates a training-evaluation mismatch: trajectories close to the logged path may still violate planning rules, while alternative trajectories farther from the demonstration can remain valid and high-scoring. The mismatch is especially limiting for proposal-selection planners, whose performance depends on both candidate-set coverage and scorer ranking quality. We propose **CLOVER**, a **C**losed-**LO**op **V**alue **E**stimation and **R**anking framework for end-to-end driving planning. CLOVER first expands single-trajectory imitation into set-level proposal coverage by constructing evaluator-filtered pseudo-expert trajectories. It then performs conservative closed-loop self-distillation: a trajectory-level scorer is fitted to true evaluator sub-scores on generated proposals, while the generator is refined toward teacher-selected top-k and vector-Pareto proposal targets with stability regularization. We also analyze when an imperfect scorer can improve the generator, showing that scorer-mediated refinement is reliable under local scorer accuracy, conservative updates, and selected-set enrichment.

## TODO

- [x] Release paper
- [x] Release inference code, scripts, and ckpt
- [ ] Release full training scripts

## Diversity Visualization

<p align="center">
  <img src="fig/diversity_visualization_appendix.png" alt="Diversity visualization appendix">
</p>

## Data and Weights

Pretrained checkpoints and release assets are available at:

- `https://github.com/WilliamXuanYu/CLOVER/releases`

Please prepare the NAVSIM-v1 dataset following the official NAVSIM layout, including:

- `export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"`
- `export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"`
- `export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"`
- `export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"`
- `export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"`
- `METRIC_CACHE_PATH/...`

For the visual backbone, download the DINOv2 ViT-S weights from:

- `https://huggingface.co/timm/vit_small_patch14_reg4_dinov2.lvd142m/tree/main`

and place them under:

- `./weights/vit_small_patch14_reg4_dinov2.lvd142m`

The default expected file is:

- `weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`

You can also override the backbone weight path with:

- `CLOVER_IMAGE_BACKBONE_WEIGHTS`

Download the released CLOVER checkpoint from:

- `https://github.com/WilliamXuanYu/CLOVER/releases`

## Installation

```bash
conda create -n clover python=3.8
conda activate clover
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e /path/to/nuplan-devkit
pip install -e .
```

If you prefer to use the vendored `nuplan-devkit` copy in this repository instead of an external checkout:

```bash
pip install -e ./nuplan-devkit
```

## Environment Variables

Before running inference, set:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/clover/dataset/maps"
export OPENSCENE_DATA_ROOT="/PATH/TO/clover/dataset"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/clover"
export NAVSIM_EXP_ROOT="/PATH/TO/clover/exp"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"
export METRIC_CACHE_PATH="/PATH/TO/clover/metric_cache"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="/PATH/TO/model.safetensors"
```

You also need to provide:

```bash
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

Before evaluation, download the released checkpoint from:

- `https://github.com/WilliamXuanYu/CLOVER/releases`

Current public evaluation entrypoint:

```bash
bash scripts/eval_multi_expert_navtest.sh
```

A complete example:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/PATH/TO/clover/dataset/maps"
export OPENSCENE_DATA_ROOT="/PATH/TO/clover/dataset"
export NAVSIM_DEVKIT_ROOT="/PATH/TO/clover"
export NAVSIM_EXP_ROOT="/PATH/TO/clover/exp"
export SUBSCORE_PATH="${NAVSIM_EXP_ROOT}"
export METRIC_CACHE_PATH="/PATH/TO/clover/metric_cache"
export CLOVER_IMAGE_BACKBONE_WEIGHTS="/PATH/TO/model.safetensors"
export CHECKPOINT="/PATH/TO/clover.ckpt"

bash scripts/eval_multi_expert_navtest.sh
```

Outputs are written to:

- `${NAVSIM_EXP_ROOT}/ke/${EXPERIMENT}/...`
- `${SUBSCORE_PATH}/navsim1_pdm_scores/${EXPERIMENT}/...`
