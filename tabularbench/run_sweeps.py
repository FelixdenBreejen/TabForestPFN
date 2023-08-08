from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
import openml
import random
import numpy as np

from tabularbench.configs.all_model_configs import total_config
from tabularbench.run_experiment import train_model_on_config
from tabularbench.random_search_object import RandomSearchObject

SWEEP_FILE_NAME = 'sweep.csv'


def run_sweeps(output_dir: str, seed: int = 0, main_process: bool = True):
    """
    Run all sweeps in the sweep.csv file in the output_dir.
    If main_process is True, then this process will also make the graphs.
    If main_process is False, then this process will only run the sweeps.
    """

    random.seed(seed)
    np.random.seed(seed)

    sweep_csv_path = Path(output_dir) / SWEEP_FILE_NAME
    sweep_csv = pd.read_csv(sweep_csv_path)

    for i, row in sweep_csv.iterrows():
        
        if row['random_search']:
            random_search_str = 'random'
        else:
            random_search_str = 'default'

        results_file_name = f"results_{row['benchmark']}_{random_search_str}_{row['model']}.csv"
        
        if row['random_search']:
            random_search_sweep(row.to_dict(), output_dir, results_file_name, main_process)
        else:
            grid_default_sweep(row.to_dict(), output_dir, results_file_name, main_process)
    




def random_search_sweep(benchmark: dict[str, str], output_dir: Path, results_file_name: str, main_process: bool):
    
    model = benchmark['model']
    task = benchmark['task']
    config = total_config[model][task]['random']
    random_search_objects = [RandomSearchObject(name, cfg) for name, cfg in config.items()]
    results_path = Path(output_dir) / results_file_name
    datasets_all_ids = openml.study.get_suite(benchmark['suite_id']).tasks
    
    while True:

        datasets_unfinished = get_unfinished_datasets(datasets_all_ids, results_path, benchmark['runs_per_dataset'])

        if len(datasets_unfinished) == 0:
            break

        
        config_base = make_base_config(benchmark)
        config_dataset = draw_dataset_config(datasets_unfinished)
        config_hyperparams = draw_random_config(random_search_objects)
        config_run = {**config_base, **config_dataset, **config_hyperparams}

        results = train_model_on_config(config_run)

        if results == -1:
            continue

        df_new = pd.Series(results).to_frame().T

        if not results_path.exists():
            df_new.to_csv(results_path, mode='w', index=False, header=True)
        else:
            df = pd.read_csv(results_path)
            df = df.append(df_new, ignore_index=True)
            df.to_csv(results_path, mode='w', index=False, header=True)
        

    
    if main_process:
        pass
        # make_graphs(output_dir)

    


def get_unfinished_datasets(datasets_all_ids: list[int], results_path: Path, runs_per_dataset: int) -> list[int]:

    if not results_path.exists():
        return datasets_all_ids
    
    results_df = pd.read_csv(results_path)
    datasets_run_count = results_df.groupby('data__keyword').count()['data__categorical'].to_dict()

    datasets_unfinished = []
    for dataset_id in datasets_all_ids:
        if dataset_id not in datasets_run_count:
            datasets_unfinished.append(dataset_id)
        elif datasets_run_count[dataset_id] < runs_per_dataset:
            datasets_unfinished.append(dataset_id)

    return datasets_unfinished


def make_base_config(benchmark: dict) -> dict:

    dataset_size = benchmark['dataset_size']

    if dataset_size == "small":
        max_train_samples = 1000
    elif dataset_size == "medium":
        max_train_samples = 10000
    elif dataset_size == "large":
        max_train_samples = 50000
    else:
        assert type(dataset_size) == int
        max_train_samples = dataset_size

    return {
        "data__categorical": benchmark['task'] == 'classif',
        "data__method_name": "openml_no_transform",
        "data__regression": benchmark['task'] == 'regression',
        "regression": benchmark['task'] == 'regression',
        "n_iter": 'auto',
        "max_train_samples": max_train_samples
    }

def draw_dataset_config(datasets_unfinished: list[int]) -> dict:

    dataset_id = random.choice(datasets_unfinished)
    return {
        "data__keyword": dataset_id
    }


def draw_random_config(random_search_objects: list[RandomSearchObject]) -> dict:

    random_config = {}

    for random_search_object in random_search_objects:
        random_config[random_search_object.name] = random_search_object.draw()

    return random_config
    

def grid_default_sweep(model: str, benchmark: str):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--main_process', action='store_true', help='Whether this is the main process (makes graphs)')

    args = parser.parse_args()

    run_sweeps(args.output_dir, seed=args.seed, main_process=args.main_process)

