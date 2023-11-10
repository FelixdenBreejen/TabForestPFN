from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch


from tabularbench.core.enums import DatasetSize, FeatureType, ModelName, SearchType, Task


@dataclass
class ConfigDatasetSweep():
    logger: logging.Logger
    output_dir: Path
    seed: int
    devices: list[torch.device]
    model_name: ModelName
    search_type: SearchType
    task: Task
    feature_type: FeatureType
    dataset_size: DatasetSize
    openml_task_id: int
    openml_dataset_id: int
    openml_dataset_name: str
    n_random_runs: int
    hyperparams_object: DictConfig
