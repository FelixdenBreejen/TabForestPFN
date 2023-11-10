from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Self
from omegaconf import DictConfig

import torch

from tabularbench.core.enums import FeatureType, ModelName, Task, DatasetSize
from tabularbench.sweeps.config_dataset_sweep import ConfigDatasetSweep


@dataclass
class ConfigRun():
    logger: logging.Logger
    device: torch.device
    seed: int
    device: torch.device
    model_name: ModelName
    task: Task
    feature_type: FeatureType
    dataset_size: DatasetSize
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    hyperparams: DictConfig


    @classmethod
    def create(cls, cfg: ConfigDatasetSweep, device: torch.device, hyperparams: DictConfig) -> Self:

        return cls(
            logger=cfg.logger,
            model_name=cfg.model_name,
            device=device,
            seed=cfg.seed,
            task=cfg.task,
            feature_type=cfg.feature_type,
            dataset_size=cfg.dataset_size,
            openml_task_id=cfg.openml_task_id,
            openml_dataset_id=cfg.openml_dataset_id,
            openml_dataset_name=cfg.openml_dataset_name,
            hyperparams=hyperparams
        )
    

    def to_results_dict(self) -> dict:

        result_dict = {
            'model': self.model,
            'device': str(self.device),
            'task': self.task.name,
            'feature_type': self.feature_type.name,
            'dataset_size': self.dataset_size.name,
            'openml_task_id': self.openml_task_id,
            'openml_dataset_id': self.openml_dataset_id,
            'openml_dataset_name': self.openml_dataset_name,
        }

        for key, value in self.hyperparams.items():
            result_dict["hp__"+str(key)] = value

        return result_dict
