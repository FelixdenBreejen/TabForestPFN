from dataclasses import dataclass
from pathlib import Path

import torch

from tabularbench.config.config_plotting import ConfigPlotting
from tabularbench.config.config_save_load_mixin import ConfigSaveLoadMixin
from tabularbench.core.enums import ModelName, SearchType
from tabularbench.data.benchmarks import Benchmark


@dataclass
class ConfigBenchmarkSweep(ConfigSaveLoadMixin):
    output_dir: Path
    seed: int
    devices: list[torch.device]
    benchmark: Benchmark
    model_name: ModelName
    model_plot_name: str
    search_type: SearchType
    plotting: ConfigPlotting
    n_random_runs_per_dataset: int
    n_default_runs_per_dataset: int
    openml_dataset_ids_to_ignore: list[int]
    hyperparams_object: dict


    def __post_init__(self):

        self.openml_dataset_ids_to_ignore = list(set(self.openml_dataset_ids_to_ignore) & set(self.benchmark.openml_dataset_ids))
        assert set(self.openml_dataset_ids_to_ignore) <= set(self.benchmark.openml_dataset_ids), f"openml_dataset_ids_to_ignore {self.openml_dataset_ids_to_ignore} contains ids that are not in benchmark {self.benchmark.name}"
        self.openml_dataset_ids_to_use = list(set(self.benchmark.openml_dataset_ids) - set(self.openml_dataset_ids_to_ignore))
        self.openml_dataset_ids_to_use.sort()







