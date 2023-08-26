from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import fcntl

import pandas as pd
import random
import numpy as np
import torch

from tabularbench.configs.all_model_configs import total_config
from tabularbench.run_experiment import train_model_on_config
from tabularbench.sweeps.random_search_object import WandbSearchObject
from tabularbench.sweeps.sweep_config import SweepConfig, sweep_config_maker
from tabularbench.sweeps.datasets import get_unfinished_task_ids
from tabularbench.sweeps.paths_and_filenames import SWEEP_FILE_NAME, RESULTS_FILE_NAME



def run_sweeps(output_dir: str, gpu: int, seed: int = 0):
    """
    Run all sweeps in the sweep.csv file in the output_dir.
    If main_process is True, then this process will also make the graphs.
    If main_process is False, then this process will only run the sweeps.
    """

    print("seed: ", seed)

    log_to_file(output_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu'


    sweep_csv = pd.read_csv(Path(output_dir) / SWEEP_FILE_NAME)
    sweep_configs = sweep_config_maker(sweep_csv, output_dir)

    for sweep_config in sweep_configs:
        
        search_sweep(sweep_config, seed=seed, device=device, is_random=False)
        if sweep_config.random_search:
            search_sweep(sweep_config, seed=seed, device=device, is_random=True)


def log_to_file(output_dir: str):

    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name_stdout = str(os.getpid()) + ".out"
    log_file_name_stderr = str(os.getpid()) + "_error.out"

    sys.stdout = open(log_dir / log_file_name_stdout, "a", 1)
    sys.stderr = open(log_dir / log_file_name_stderr, "a", 1)


def search_sweep(sweep: SweepConfig, seed: int, device: str, is_random: bool):
    """Perform one sweep: one row of the sweep.csv file."""
    
    config = total_config[sweep.model][sweep.task]
    search_object = WandbSearchObject(config)
    results_path = sweep.path / RESULTS_FILE_NAME
    runs_per_dataset = sweep.runs_per_dataset if is_random else 1
    
    while True:

        datasets_unfinished = get_unfinished_task_ids(sweep.task_ids, results_path, runs_per_dataset)

        if len(datasets_unfinished) == 0:
            break
        
        config_run = create_run_config(sweep, datasets_unfinished, search_object, seed, device, is_random)
        results = train_model_on_config(config_run)

        if results == -1:
            # This is the error code in case the run crashes
            continue

        if config_run['data__keyword'] not in get_unfinished_task_ids(sweep.task_ids, results_path, runs_per_dataset):
            # This is the case where another process finished the dataset while this process was running
            # It is important to check this because otherwise the results default runs will be saved multiple times,
            # which is problematic for computing random search statistics.
            continue

        save_results(results, results_path)


def create_run_config(
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
        "data__categorical": sweep.task == 'classif',
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
    

def save_results(results: dict, results_path: Path):

    df_new = pd.Series(results).to_frame().T

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(results_path, mode='w', index=False, header=True)
    else:
        df = pd.read_csv(results_path)
        df = df.append(df_new, ignore_index=True)
        df.to_csv(results_path, mode='w', index=False, header=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--main_process', action='store_true', help='Whether this is the main process (makes graphs)')

    args = parser.parse_args()

    run_sweeps(args.output_dir, seed=args.seed, main_process=args.main_process)

