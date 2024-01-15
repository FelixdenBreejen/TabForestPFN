from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Self
from omegaconf import DictConfig

import torch

from tabularbench.core.enums import ModelName, Task, DatasetSize
from tabularbench.data.datafile_openml import OpenmlDatafile
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.get_logger import get_logger


@dataclass
class ConfigRun():
    logger: logging.Logger
    output_dir: Path
    device: torch.device
    seed: int
    device: torch.device
    model_name: ModelName
    task: Task
    dataset_size: DatasetSize
    openml_dataset_id: int
    openml_dataset_name: str
    hyperparams: DictConfig


    @classmethod
    def create(
            cls, 
            cfg: ConfigBenchmarkSweep, 
            seed: int,
            device: torch.device, 
            dataset_id: int, 
            hyperparams: DictConfig,
            run_id: int
        ) -> Self:

        dataset_size = cfg.benchmark.dataset_size
        openml_dataset_name = OpenmlDatafile(dataset_id, dataset_size).ds.attrs['openml_dataset_name']
        
        output_dir = cfg.output_dir / str(dataset_id) / f"#{run_id}"
        logger = get_logger(output_dir / 'log.txt')

        return cls(
            logger=logger,
            output_dir=output_dir,
            model_name=cfg.model_name,
            device=device,
            seed=seed,
            task=cfg.benchmark.task,
            dataset_size=dataset_size,
            openml_dataset_id=dataset_id,
            openml_dataset_name=openml_dataset_name,
            hyperparams=hyperparams
        )
    

    def to_results_dict(self) -> dict:

        result_dict = {
            'model': self.model,
            'device': str(self.device),
            'task': self.task.name,
            'dataset_size': self.dataset_size.name,
            'openml_task_id': self.openml_task_id,
            'openml_dataset_id': self.openml_dataset_id,
            'openml_dataset_name': self.openml_dataset_name,
        }

        for key, value in self.hyperparams.items():
            result_dict["hp__"+str(key)] = value

        return result_dict
