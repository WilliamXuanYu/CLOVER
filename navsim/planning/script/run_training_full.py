import os
import random
from typing import Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import torch.distributed as dist
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def dist_ready():
    return dist.is_available() and dist.is_initialized()


def _get_eval_log_names(cfg: DictConfig):
    """Return validation log names, optionally using test logs for new reranker runs."""
    if bool(cfg.get("use_test_logs_for_val", False)):
        return cfg.test_logs
    return cfg.val_logs


def _get_eval_split_name(cfg: DictConfig) -> str:
    """Return which split should back the validation dataset."""
    if bool(cfg.get("use_test_logs_for_val", False)):
        # In this repo, labeled held-out evaluation uses the `navtest` scene
        # filter/config while the underlying files live under the `test/`
        # dataset root. Do not map this to scene_filter/test.yaml, which does
        # not exist here.
        return "navtest"
    return str(getattr(cfg.train_test_split, "data_split", cfg.get("split", "trainval")))


def _get_eval_dataset_paths(cfg: DictConfig) -> Tuple[Path, Path]:
    """
    Return dataset paths for validation.

    By default validation uses the same split root as training (e.g. trainval).
    For special reranker experiments we optionally validate on the test split, so
    both navsim_log_path and sensor_blobs_path must be remapped from
    `.../<train_split>` to `.../test`.
    """
    if not bool(cfg.get("use_test_logs_for_val", False)):
        return Path(cfg.navsim_log_path), Path(cfg.sensor_blobs_path)

    openscene_root = Path(os.getenv("OPENSCENE_DATA_ROOT", ""))
    if not openscene_root:
        raise RuntimeError("OPENSCENE_DATA_ROOT must be set when use_test_logs_for_val=true")

    return (
        openscene_root / "navsim_logs" / "test",
        openscene_root / "sensor_blobs" / "test",
    )


def _build_eval_scene_filter(cfg: DictConfig) -> SceneFilter:
    """
    Build the validation scene filter.

    When use_test_logs_for_val=true, we must not reuse the navtrain scene filter:
    it contains navtrain token lists, which would filter the test split down to
    zero samples. In that mode we explicitly load the navtest scene filter.
    """
    if not bool(cfg.get("use_test_logs_for_val", False)):
        return instantiate(cfg.train_test_split.scene_filter)

    eval_split_name = _get_eval_split_name(cfg)
    eval_scene_filter_cfg = OmegaConf.load(
        Path(__file__).resolve().parent / "config" / "common" / "train_test_split" / "scene_filter" / f"{eval_split_name}.yaml"
    )
    return instantiate(eval_scene_filter_cfg)

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs or log_name in cfg.val_logs 
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs + cfg.val_logs
    

    print("len(train_scene_filter.log_names) ", len(train_scene_filter.log_names))

    use_test_logs_for_val = bool(cfg.get("use_test_logs_for_val", False))
    eval_logs = _get_eval_log_names(cfg)
    val_scene_filter: SceneFilter = _build_eval_scene_filter(cfg)
    if use_test_logs_for_val:
        # Match run_pdm_score_multi_gpu (train_test_split=navtest): use the full
        # navtest scene list from scene_filter/navtest.yaml. Intersecting with
        # cfg.test_logs would shrink val to a small subset and make val/top1_score
        # incomparable to offline CSV means (~0.93 on full navtest).
        if val_scene_filter.log_names is not None:
            logger.info(
                "use_test_logs_for_val: validating on full navtest (%d logs), not cfg.test_logs.",
                len(val_scene_filter.log_names),
            )
    elif val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in eval_logs]
    else:
        val_scene_filter.log_names = eval_logs

    train_data_path = Path(cfg.navsim_log_path)
    train_sensor_blobs_path = Path(cfg.sensor_blobs_path)
    val_data_path, val_sensor_blobs_path = _get_eval_dataset_paths(cfg)

    logger.info("Train data path: %s", train_data_path)
    logger.info("Val data path: %s", val_data_path)
    logger.info("Eval split name: %s", _get_eval_split_name(cfg))

    train_scene_loader = SceneLoader(
        sensor_blobs_path=train_sensor_blobs_path,
        data_path=train_data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=val_sensor_blobs_path,
        data_path=val_data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        if bool(cfg.get("use_test_logs_for_val", False)):
            navtest_sf_path = (
                Path(__file__).resolve().parent
                / "config"
                / "common"
                / "train_test_split"
                / "scene_filter"
                / "navtest.yaml"
            )
            navtest_sf_cfg = OmegaConf.load(navtest_sf_path)
            eval_logs = list(OmegaConf.to_object(navtest_sf_cfg.log_names))
            logger.info(
                "use_test_logs_for_val + cache-only val: using %d navtest log folders from %s",
                len(eval_logs),
                navtest_sf_path.name,
            )
        else:
            eval_logs = _get_eval_log_names(cfg)
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=eval_logs,
        )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    # Build dataloader params without drop_last to avoid duplicate keyword (config may set it)
    train_dloader_params = {k: v for k, v in cfg.dataloader.params.items() if k != "drop_last"}
    train_dataloader = DataLoader(
        train_data, **train_dloader_params, shuffle=True, drop_last=True
    )
    logger.info("Num training samples: %d", len(train_data))
    # Validation should never drop the tail batch. When the held-out split is
    # small (especially if we use test logs as val under DDP), drop_last=True
    # can silently reduce validation to zero steps, which also prevents
    # val/score_epoch from being logged and disables best.ckpt selection.
    val_dloader_params = {k: v for k, v in cfg.dataloader.params.items() if k != "drop_last"}
    val_dataloader = DataLoader(
        val_data, **val_dloader_params, shuffle=False, drop_last=False
    )
    logger.info("Num validation samples: %d", len(val_data))
    logger.info(
        "Validation dataloader batches: %d (drop_last=%s, use_test_logs_for_val=%s)",
        len(val_dataloader),
        False,
        bool(cfg.get("use_test_logs_for_val", False)),
    )

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    auto_resume_ckpt = bool(cfg.get("auto_resume_ckpt", True))
    if auto_resume_ckpt and cfg.train_ckpt_path is None:
        # Only resume from the current experiment directory. Searching sibling
        # experiment folders can pick up checkpoints from different agent
        # architectures (e.g. an older reranker), which then crashes during
        # Lightning state restoration with incompatible keys.
        search_pattern = str(Path(cfg.output_dir) / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt")
        print(str(cfg.output_dir))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)
    elif not auto_resume_ckpt:
        cfg.train_ckpt_path = None
        print("Auto-resume disabled: start training from scratch.")

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        subscore_root = (
            os.getenv("SUBSCORE_PATH")
            or os.getenv("NAVSIM_EXP_ROOT")
            or str(Path(cfg.output_dir).resolve().parent)
        )
        dump_root = os.path.join(subscore_root, "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        if hasattr(val_data, "append_token_to_batch"):
            val_data.append_token_to_batch = True
        predict_dataloader = DataLoader(
            val_data, **val_dloader_params, shuffle=False, drop_last=False
        )
        predictions = trainer.predict(
            AgentLightningModule(agent=agent, for_viz=True),
            predict_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(merged_predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()
