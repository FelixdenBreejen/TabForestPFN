from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch


from tabularbench.core.enums import DatasetSize, FeatureType, ModelName, SearchType, Task
from tabularbench.sweeps.sweep_start import get_logger


@dataclass
class ConfigDatasetSweep():
    logger: logging.Logger
    output_dir: Path
    seed: int
    devices: list[torch.device]
    model_name: ModelName
    task: Task
    feature_type: FeatureType
    dataset_size: DatasetSize
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    n_random_runs: int
    hyperparams: DictConfig

    @classmethod
    def from_hydra(
        cls, 
        cfg_hydra: DictConfig, 
        output_dir_bench: Path, 
        model: ModelName, 
        search_type: SearchType, 
        hyperparam_configs: DictConfig
    ):

        output_dir = output_dir_bench / f'{openml_dataset_id}'
        logger = get_logger(output_dir / 'log.txt')

        logger.info(f"Start creating dataset sweep config for {model.name}-{search_type.name}")

        return cls(
            logger=logger,
            output_dir=output_dir,
            seed=cfg_hydra.seed,
            model=model,
            search_type=search_type,
            hyperparam_configs=hyperparam_configs,
        )
