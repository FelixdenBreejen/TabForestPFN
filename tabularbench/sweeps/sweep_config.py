from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from tabularbench.data.benchmarks import benchmark_names


@dataclass
class SweepConfig():
    model: str
    plot_name: str
    task: str
    feature_type: str     # what the paper calls 'categorical', we call 'mixed'
    benchmark_name: str
    search_type: str
    runs_per_dataset: int
    dataset_size: str


    def __post_init__(self):

        # TODO: validate model name
        assert self.benchmark_name in benchmark_names, f"{self.benchmark_name} is not a valid benchmark. Please choose from {benchmark_names}"
        assert self.dataset_size in [10000, 50000]
        assert self.search_type in ['default', 'random'], f"{self.search_type} is not a valid search type. Please choose from ['default', 'random']"
        assert self.task in ['regression', 'classification'], f"{self.task} is not a valid task. Please choose from ['regression', 'classif']"
        assert self.feature_type in ['numerical', 'categorical', 'mixed'], f"{self.feature_type} is not a valid feature type. Please choose from ['numerical', 'categorical', 'mixed']"

        self.folder_name = f'{self.benchmark_name}_{self.search_type}_{self.model}'



    @classmethod
    def from_dict(cls, sweep_dict: dict) -> SweepConfig:

        return cls(
            model=sweep_dict['model'],
            plot_name=sweep_dict['plot_name'],
            task=sweep_dict['task'],
            feature_type=sweep_dict['feature_type'],
            benchmark_name=sweep_dict['benchmark_name'],
            search_type=sweep_dict['search_type'],
            runs_per_dataset=sweep_dict['runs_per_dataset'],
            dataset_size=sweep_dict['dataset_size'],
        )
    

    def to_dict(self) -> dict:

        return {
            'model': self.model,
            'plot_name': self.plot_name,
            'task': self.task,
            'feature_type': self.feature_type,
            'benchmark_name': self.benchmark_name,
            'search_type': self.search_type,
            'runs_per_dataset': self.runs_per_dataset,
            'dataset_size': self.dataset_size,
        }
    

    def to_dict_all_params(self) -> dict:

        return {
            'model': self.model,
            'plot_name': self.plot_name,
            'task': self.task,
            'feature_type': self.feature_type,
            'benchmark_name': self.benchmark_name,
            'search_type': self.search_type,
            'runs_per_dataset': self.runs_per_dataset,
            'dataset_size': self.dataset_size,
            'folder_name': self.folder_name,
        }
    

def save_sweep_config_list_to_file(sweep_configs: list[SweepConfig], path: Path) -> None:

    sweep_dicts = [sweep_config.to_dict() for sweep_config in sweep_configs]
    pd.DataFrame(sweep_dicts).to_csv(path, index=False)


def load_sweep_configs_from_file(path: Path) -> list[SweepConfig]:

    sweep_dicts = pd.read_csv(path).to_dict('records')
    sweep_configs = [SweepConfig.from_dict(sweep_dict) for sweep_dict in sweep_dicts]

    return sweep_configs