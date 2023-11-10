from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from omegaconf import DictConfig
import logging

import torch

from tabularbench.sweeps.writer import Writer
from tabularbench.core.enums import DatasetSize, ModelName, Task, FeatureType, SearchType


@dataclass
class SweepConfig():
    logger: logging.Logger
    writer: Writer
    output_dir: Path
    seed: int
    device: Optional[torch.device]
    model: ModelName
    model_plot_name: str
    task: Task
    feature_type: FeatureType
    benchmark_name: str
    search_type: SearchType
    openml_suite_id: int
    openml_task_ids: list[int]
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]
    runs_per_dataset: int
    dataset_size: DatasetSize
    hyperparams: DictConfig      # hyperparameters for the model
    plotting: DictConfig         # plotting parameters


    def __post_init__(self):

        assert self.benchmark_name in benchmark_names, f"{self.benchmark_name} is not a valid benchmark. Please choose from {benchmark_names}"
        self.sweep_dir = self.output_dir / f'{self.benchmark_name}_{self.model.name}_{self.search_type.name}'


    def __str__(self) -> str:

        return f"{self.benchmark_name}-{self.model.name}-{self.search_type.name}"
    

