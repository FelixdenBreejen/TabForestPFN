from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch


from tabularbench.core.enums import ModelName, SearchType
from tabularbench.data.benchmarks import Benchmark



@dataclass
class ConfigBenchmarkSweep():
    logger: logging.Logger
    output_dir: Path
    seed: int
    devices: list[torch.device]
    benchmark: Benchmark
    model_name: ModelName
    model_plot_name: str
    search_type: SearchType
    config_plotting: ConfigPlotting
    n_random_runs_per_dataset: int
    n_default_runs_per_dataset: int
    openml_dataset_ids_to_ignore: list[int]
    hyperparams_object: DictConfig


    def __post_init__(self):

        assert set(self.openml_dataset_ids_to_ignore) <= set(self.benchmark.openml_dataset_ids), f"openml_dataset_ids_to_ignore {self.openml_dataset_ids_to_ignore} contains ids that are not in benchmark {self.benchmark.name}"
        self.openml_dataset_ids_to_use = list(set(self.benchmark.openml_dataset_ids) - set(self.openml_dataset_ids_to_ignore))



@dataclass
class ConfigPlotting():
    n_runs: int
    n_random_shuffles: int
    confidence_bound: float
    benchmark_model_names: list[ModelName]



