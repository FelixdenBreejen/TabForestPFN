from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
import itertools
import logging

import torch
import openml
import pandas as pd

from tabularbench.data.benchmarks import benchmarks, benchmark_names
from tabularbench.sweeps.writer import Writer


@dataclass
class SweepConfig():
    logger: logging.Logger
    writer: Writer
    output_dir: Path
    seed: int
    device: torch.device
    model: str
    model_plot_name: str
    task: str
    feature_type: str     # what the paper calls 'categorical', we call 'mixed'
    benchmark_name: str
    search_type: str
    openml_suite_id: int
    openml_task_ids: list[int]
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]
    runs_per_dataset: int
    dataset_size: int
    model_hyperparameters: DictConfig      # hyperparameters for the model

    def __post_init__(self):

        # TODO: validate model name
        assert self.benchmark_name in benchmark_names, f"{self.benchmark_name} is not a valid benchmark. Please choose from {benchmark_names}"
        assert self.dataset_size in [10000, 50000]
        assert self.search_type in ['default', 'random'], f"{self.search_type} is not a valid search type. Please choose from ['default', 'random']"
        assert self.task in ['regression', 'classification'], f"{self.task} is not a valid task. Please choose from ['regression', 'classification']"
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
    

    def __str__(self) -> str:

        return f"{self.benchmark_name}-{self.search_type}-{self.model}"
    

def create_sweep_config_list_from_main_config(cfg: DictConfig, writer: Writer, logger: logging.Logger) -> list[SweepConfig]:

    sweep_configs = []

    assert len(cfg.models) == len(cfg.model_plot_names), f"Please provide a plot name for each model. Got {len(cfg.models)} models and {len(cfg.model_plot_names)} plot names."

    models_with_plot_name = zip(cfg.models, cfg.model_plot_names)
    sweep_details = itertools.product(models_with_plot_name, cfg.search_type, cfg.benchmarks)

    for (model, model_plot_name), search_type, benchmark_name in sweep_details:

        for b in benchmarks:
            if b['name'] == benchmark_name:
                benchmark = b
                break
        else:
            raise ValueError(f"Benchmark {benchmark_name} not found in benchmarks")

        if search_type == 'default':
            runs_per_dataset = 1
        else:
            runs_per_dataset = cfg.runs_per_dataset

        if benchmark['categorical']:
            feature_type = 'mixed'
        else:
            feature_type = 'numerical'

        if benchmark['task'] == 'regression':
            task = 'regression'
        elif benchmark['task'] == 'classif':
            task = 'classification'
        else:
            raise ValueError(f"task must be one of ['regression', 'classif']. Got {benchmark['task']}")

            
        if benchmark['dataset_size'] == 'small':
            dataset_size = 1000
        elif benchmark['dataset_size'] == 'medium':
            dataset_size = 10000
        elif benchmark['dataset_size'] == 'large':
            dataset_size = 50000
        else:
            raise ValueError(f"dataset_size_str must be one of ['small', 'medium', 'large']. Got {benchmark['dataset_size']}")
        
        assert model in cfg.hyperparams, f"Model {model} not found in main configuration's hyperparams"

        openml_suite = openml.study.get_suite(benchmark['suite_id'])
        openml_task_ids = openml_suite.tasks
        openml_dataset_ids = openml_suite.data
        openml_dataset_names = [openml.datasets.get_dataset(dataset_id).name for dataset_id in openml_dataset_ids]
        
        sweep_config = SweepConfig(
            logger=logger,
            writer=writer,
            output_dir=Path(cfg.output_dir),
            seed=cfg.seed,
            device=torch.device(cfg.device),
            model=model,
            model_plot_name=model_plot_name,
            benchmark_name=benchmark['name'],
            search_type=search_type,
            task=task,
            dataset_size=dataset_size,
            feature_type=feature_type,
            openml_suite_id=benchmark['suite_id'],
            openml_task_ids=openml_task_ids,
            openml_dataset_ids=openml_dataset_ids,
            openml_dataset_names=openml_dataset_names,
            runs_per_dataset=runs_per_dataset,
            model_hyperparameters=cfg.hyperparams[model]
        )

        sweep_configs.append(sweep_config)

    return sweep_configs
