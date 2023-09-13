from __future__ import annotations
from pathlib import Path
import openml
from dataclasses import dataclass
import logging
from omegaconf import DictConfig

import torch
import pandas as pd

from tabularbench.data.benchmarks import benchmark_names
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.writer import Writer


@dataclass
class RunConfig():
    logger: logging.Logger
    writer: Writer
    device: torch.device
    model: str
    seed: int
    task: str
    feature_type: str
    dataset_size: int
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    model_hyperparameters: DictConfig


    def __post_init__(self):

        assert self.dataset_size in [10000, 50000]
        assert self.task in ['regression', 'classification'], f"{self.task} is not a valid task. Please choose from ['regression', 'classification']"
        assert self.feature_type in ['numerical', 'categorical', 'mixed'], f"{self.feature_type} is not a valid feature type. Please choose from ['numerical', 'categorical', 'mixed']"



    @classmethod
    def create(cls, sweep_cfg: SweepConfig, dataset_id: int, hyperparams: DictConfig) -> RunConfig:

        openml_index = sweep_cfg.openml_dataset_ids.index(dataset_id)
        task_id = sweep_cfg.openml_task_ids[openml_index]
        dataset_name = sweep_cfg.openml_dataset_names[openml_index]

        return cls(
            logger=sweep_cfg.logger,
            writer=sweep_cfg.writer,
            model=sweep_cfg.model,
            device=sweep_cfg.device,
            seed=sweep_cfg.seed,
            task=sweep_cfg.task,
            feature_type=sweep_cfg.feature_type,
            dataset_size=sweep_cfg.dataset_size,
            openml_task_id=task_id,
            openml_dataset_id=dataset_id,
            openml_dataset_name=dataset_name,
            model_hyperparameters=hyperparams
        )
