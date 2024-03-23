from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import DictConfig

from tabularbench.config.config_run import ConfigRun
from tabularbench.core.enums import DatasetSize, ModelName, SearchType, Task
from tabularbench.results.run_metrics import RunMetrics


@dataclass
class ResultsRun():
    model_name: ModelName
    openml_dataset_id: int
    openml_dataset_name: str
    task: Task
    dataset_size: DatasetSize
    search_type: SearchType
    seed: int
    device: Optional[torch.device]
    metrics: RunMetrics
    hyperparams: DictConfig


    # def to_dict(self):

    #     d = {
    #         'model': self.model_name.name,
    #         'openml_dataset_id': self.openml_dataset_id,
    #         'openml_dataset_name': self.openml_dataset_name,
    #         'task': self.task.name,
    #         'dataset_size': self.dataset_size.name,
    #         'search_type': self.search_type.name,
    #         'seed': self.seed,
    #         'device': str(self.device),
    #     }

    #     for key, value in self.hyperparams.items():
    #         d["hp__"+str(key)] = value

    #     return d
    

    @classmethod
    def from_run_config(
        cls,
        cfg: ConfigRun, 
        search_type: SearchType,
        metrics: RunMetrics
    ) -> ResultsRun:

        return cls(
            model_name=cfg.model_name,
            openml_dataset_id=cfg.openml_dataset_id,
            openml_dataset_name=cfg.openml_dataset_name,
            task=cfg.task,
            dataset_size=cfg.dataset_size,
            search_type=search_type,
            seed=cfg.seed,
            device=cfg.device,
            metrics=metrics,
            hyperparams=cfg.hyperparams,
        )



