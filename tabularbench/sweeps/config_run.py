from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Self
from omegaconf import DictConfig

import torch

from tabularbench.core.enums import ModelName, Task, DatasetSize
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.sweep_start import get_logger


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
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    hyperparams: DictConfig


    @classmethod
    def create(
            cls, 
            cfg: ConfigBenchmarkSweep, 
            device: torch.device, 
            dataset_id: int, 
            hyperparams: DictConfig,
            run_id: int
        ) -> Self:

        id_index = cfg.benchmark.openml_dataset_ids.index(dataset_id)
        openml_task_id = cfg.benchmark.openml_task_ids[id_index]
        openml_dataset_name = cfg.benchmark.openml_dataset_names[id_index]
        
        output_dir = cfg.output_dir / str(dataset_id) / f"#{run_id}"
        logger = get_logger(output_dir / 'log.txt')

        return cls(
            logger=logger,
            output_dir=output_dir,
            model_name=cfg.model_name,
            device=device,
            seed=cfg.seed,
            task=cfg.benchmark.task,
            dataset_size=cfg.benchmark.dataset_size,
            openml_task_id=openml_task_id,
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
