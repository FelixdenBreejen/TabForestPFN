from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Generator

from omegaconf import DictConfig
import torch


from tabularbench.core.enums import ModelName, SearchType
from tabularbench.data.benchmarks import Benchmark
from tabularbench.sweeps.config_dataset_sweep import ConfigDatasetSweep
from tabularbench.sweeps.sweep_start import get_logger



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
    openml_dataset_ids_to_ignore: list[int]
    hyperparams: DictConfig
    
    
    def generate_configs_dataset_sweep(self) -> Generator[ConfigDatasetSweep, None, None]:

        for i in range(len(self.benchmark.openml_dataset_ids)):

            openml_task_id = self.benchmark.openml_task_ids[i]
            openml_dataset_id = self.benchmark.openml_dataset_ids[i]
            openml_dataset_name = self.benchmark.openml_dataset_names[i]

            if openml_dataset_id in self.openml_dataset_ids_to_ignore:
                self.logger.info(f"Ignoring dataset {openml_dataset_id} because it is in ignore_datasets (config)")
                continue

            output_dir_dataset = self.output_dir / f'{openml_dataset_id}'
            logger_dataset = get_logger(output_dir_dataset / 'log.txt')

            dataset_sweep_config = ConfigDatasetSweep(
                logger=logger_dataset,
                output_dir=output_dir_dataset,
                seed=self.seed,
                devices=self.devices,
                model_name=self.model_name,
                task=self.benchmark.task,
                feature_type=self.benchmark.feature_type,
                dataset_size=self.benchmark.dataset_size,
                openml_task_id=openml_task_id,
                openml_dataset_id=openml_dataset_id,
                openml_dataset_name=openml_dataset_name,
                n_random_runs=self.n_random_runs_per_dataset,
                hyperparams=self.hyperparams
            )

            yield dataset_sweep_config



@dataclass
class ConfigPlotting():
    n_runs: int
    n_random_shuffles: int
    confidence_bound: float
    benchmark_models: list[ModelName]



