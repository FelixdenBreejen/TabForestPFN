from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import os
import time

import pandas as pd
import openml
import random
import numpy as np

from tabularbench.configs.all_model_configs import total_config
from tabularbench.run_experiment import train_model_on_config
from tabularbench.sweeps.random_search_object import WandbSearchObject
from tabularbench.sweeps.run_sweeps import (
    sweep_generator, get_unfinished_datasets, SWEEP_FILE_NAME, RESULTS_FILE_NAME, RESULTS_MODIFIED_FILE_NAME, PATH_TO_ALL_BENCH_CSV
)


def monitor_and_make_plots(output_dir: str, delay_in_seconds: int = 10):

    sweep_csv = pd.read_csv(Path(output_dir) / SWEEP_FILE_NAME)
    sweeps = sweep_generator(sweep_csv, output_dir)

    while True:
        time.sleep(delay_in_seconds)
        for sweep in sweeps:
            if sweep_grid_finished(sweep):
                make_grid_results(sweep)

            
            make_sweep_plots(sweep, output_dir)


def sweep_grid_finished(sweep: dict[str, str]) -> bool:
    
    # default sweep always finishes before random sweep starts, so we just check if every dataset has one run
    unfinished_datasets = get_unfinished_datasets(sweep, sweep['path'] / RESULTS_FILE_NAME, runs_per_dataset=1)

    return len(unfinished_datasets) == 0


def make_grid_results(sweep: dict[str, str]):

    df = pd.read_csv(sweep['path'] / RESULTS_FILE_NAME)
    df_all = pd.read_csv(PATH_TO_ALL_BENCH_CSV)
    datasets_all_ids = openml.study.get_suite(sweep['suite_id']).tasks

    index = df_all['model_name'].unique().tolist() + [sweep['plot_name']]
    df_new = pd.DataFrame(columns=datasets_all_ids, index=index)

    df_new.loc[sweep['plot_name']] = df[df['hp'] == 'default']['mean_test_score'].to_list()

    for model_name in df_all['model_name'].unique():

        correct_model = df_all['model_name'] == model_name
        correct_task = df_all['hp'] == 'default'
        correct_benchmark = df_all['benchmark'] == sweep['benchmark'] + '_' + sweep['dataset_size']
        df_new.loc[model_name] = df_all.loc[correct_model & correct_task & correct_benchmark, 'mean_test_score'].tolist()

    id_to_name = {}
    for id in datasets_all_ids:
        dataset_id_real = openml.tasks.get_task(id).dataset_id
        dataset_name = openml.datasets.get_dataset(dataset_id_real, download_data=False).name
        id_to_name[id] = dataset_name
    
    df_new.rename(columns=id_to_name, inplace=True)
    df_new.to_csv(sweep['path'] / 'grid_results.csv', mode='w', index=True, header=True)


def make_sweep_plots(sweep: dict[str, str], output_dir: str):

    model = sweep['model']
    task = sweep['task']
    config = total_config[model][task]
    search_object = WandbSearchObject(config)
    results_path = sweep['path'] / RESULTS_FILE_NAME
    datasets_all_ids = openml.study.get_suite(sweep['suite_id']).tasks
    

    df = pd.read_csv(results_path)
    
    
    if sweep['random_search']:

        df['benchmark'] = benchmark['benchmark'] + '_' + benchmark['dataset_size']

        df['data__openmlid'] = df['data__keyword']

        for id in datasets_all_ids:
            dataset_id_real = openml.tasks.get_task(id).dataset_id
            dataset_name = openml.datasets.get_dataset(dataset_id_real, download_data=False).name

            df.loc[df['data__openmlid'] == id, 'data__keyword'] = dataset_name

        df['model_name'] = benchmark['plot_name']

        df.to_csv(benchmark_dir / RESULTS_MODIFIED_FILE_NAME, mode='w', index=False, header=True)

        script_name = f"bench_script_{benchmark['benchmark']}"
        script_path = 'analyses/' + script_name + '.R'
        results_csv_path = str(benchmark_dir)
        subprocess.run(['Rscript', script_path, results_csv_path])

    

    




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--delay_in_seconds', type=int, default=10, help='Delay between checking if sweep is finished')

    args = parser.parse_args()

    monitor_and_make_plots(args.output_dir, args.delay_in_seconds)

