from __future__ import annotations
from pathlib import Path
import openml
from dataclasses import dataclass
import logging

import pandas as pd

from tabularbench.data.benchmarks import benchmark_names
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.writer import Writer


@dataclass
class RunConfig():
    logger: logging.Logger
    writer: Writer
    model: str
    seed: int
    hp: str
    openml_id: int

    



    def __post_init__(self):
        pass



    @classmethod
    def create(cls, sweep_cfg: SweepConfig, is_random: bool) -> RunConfig:

        return cls(
            model=sweep_cfg['model'],
            seed=cfg.seed,
            hp=sweep_cfg['hp'],
            openml_id=sweep_cfg['data__keyword']
        )






def create_run_config(
    cfg: dict,
    sweep: SweepConfig, 
    datasets_unfinished: list[int], 
    search_object: WandbSearchObject,  
    seed: int,
    device: str,
    is_random: bool
) -> dict:


    config_base = make_base_config(sweep)
    config_dataset = draw_dataset_config(datasets_unfinished)
    config_hyperparams = search_object.draw_config(type='random' if is_random else 'default')
    config_hp = {'hp': 'random' if is_random else 'default', 'seed': seed}
    config_device = {'model__device': device}
    config_run = {**config_base, **config_dataset, **config_hyperparams, **config_hp, **config_device}

    return config_run


def make_base_config(sweep: SweepConfig) -> dict:

    if sweep.dataset_size == "small":
        max_train_samples = 1000
    elif sweep.dataset_size == "medium":
        max_train_samples = 10000
    elif sweep.dataset_size == "large":
        max_train_samples = 50000
    else:
        assert type(sweep.dataset_size) == int
        max_train_samples = sweep.dataset_size

    return {
        "data__categorical": sweep.categorical,
        "data__method_name": "openml_no_transform",
        "data__regression": sweep.task == 'regression',
        "regression": sweep.task == 'regression',
        "n_iter": 'auto',
        "max_train_samples": max_train_samples
    }


def draw_dataset_config(datasets_unfinished: list[int]) -> dict:

    dataset_id = random.choice(datasets_unfinished)
    return {
        "data__keyword": dataset_id
    }