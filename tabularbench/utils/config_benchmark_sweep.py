from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import torch
import yaml

from tabularbench.core.enums import ModelName, SearchType
from tabularbench.data.benchmarks import Benchmark



@dataclass
class ConfigBenchmarkSweep():
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

        self.openml_dataset_ids_to_ignore = list(set(self.openml_dataset_ids_to_ignore) & set(self.benchmark.openml_dataset_ids))
        assert set(self.openml_dataset_ids_to_ignore) <= set(self.benchmark.openml_dataset_ids), f"openml_dataset_ids_to_ignore {self.openml_dataset_ids_to_ignore} contains ids that are not in benchmark {self.benchmark.name}"
        self.openml_dataset_ids_to_use = list(set(self.benchmark.openml_dataset_ids) - set(self.openml_dataset_ids_to_ignore))
        self.openml_dataset_ids_to_use.sort()


    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # make sure the hyperparams are saved as a dict, not as an OmegaConf object
        # OmegaConf object looks ugly when saved as yaml
        hyperparams_dict = OmegaConf.to_container(self.hyperparams_object, resolve=True)
        self_to_save = dataclasses.replace(self, hyperparams_object=hyperparams_dict)
    
        with open(self.output_dir / "config_benchmark_sweep.yaml", 'w') as f:
            yaml.dump(self_to_save, f, default_flow_style=False)


@dataclass
class ConfigPlotting():
    n_runs: int
    n_random_shuffles: int
    confidence_bound: float
    plot_default_value: bool
    benchmark_model_names: list[ModelName]



