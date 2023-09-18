from __future__ import annotations
from dataclasses import dataclass
import logging
from omegaconf import DictConfig

import torch

from tabularbench.core.enums import FeatureType, ModelName, Task, DatasetSize
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.writer import Writer


@dataclass
class RunConfig():
    logger: logging.Logger
    writer: Writer
    device: torch.device
    model: ModelName
    seed: int
    task: Task
    feature_type: FeatureType
    dataset_size: DatasetSize
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    hyperparams: DictConfig


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
