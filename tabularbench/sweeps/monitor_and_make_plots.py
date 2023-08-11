from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import time

import pandas as pd

from tabularbench.sweeps.sweep_config import SweepConfig, sweep_config_maker
from tabularbench.sweeps.datasets import get_unfinished_task_ids
from tabularbench.sweeps.paths_and_filenames import (
    SWEEP_FILE_NAME, RESULTS_FILE_NAME, RESULTS_MODIFIED_FILE_NAME, 
    PATH_TO_ALL_BENCH_CSV, DEFAULT_RESULTS_FILE_NAME
)


def monitor_and_make_plots(output_dir: str, delay_in_seconds: int = 10):

    sweep_csv = pd.read_csv(Path(output_dir) / SWEEP_FILE_NAME)
    sweeps = sweep_config_maker(sweep_csv, output_dir)

    for sweep in sweeps:

        while True:
            if sweep_default_finished(sweep):
                make_default_results(sweep)
                break
            
            time.sleep(delay_in_seconds)

        while True:
            make_random_sweep_plots(sweep)

            if sweep_random_finished(sweep) or not sweep.random_search:
                break

            time.sleep(delay_in_seconds)


def sweep_default_finished(sweep: SweepConfig) -> bool:
    
    # default sweep always finishes before random sweep starts, so we just check if every dataset has one run
    unfinished_tasks = get_unfinished_task_ids(sweep.task_ids, sweep.path / RESULTS_FILE_NAME, runs_per_dataset=1)

    return len(unfinished_tasks) == 0


def sweep_random_finished(sweep: SweepConfig) -> bool:

    if not sweep.random_search:
        return True
    
    # default sweep always finishes before random sweep starts, so we just check if every dataset has one run
    unfinished_tasks = get_unfinished_task_ids(sweep.task_ids, sweep.path / RESULTS_FILE_NAME, runs_per_dataset=sweep.runs_per_dataset)

    return len(unfinished_tasks) == 0


def make_default_results(sweep: SweepConfig):

    df = pd.read_csv(sweep.path / RESULTS_FILE_NAME)
    df['data__keyword'] = df['data__keyword'].map(dict(zip(sweep.task_ids, sweep.dataset_names)))
    df.sort_values(by='data__keyword', inplace=True, ascending=True)

    df_all = pd.read_csv(PATH_TO_ALL_BENCH_CSV)

    index = df_all['model_name'].unique().tolist() + [sweep.plot_name]
    df_new = pd.DataFrame(columns=df['data__keyword'].unique().tolist(), index=index)

    df_new.loc[sweep.plot_name] = df[df['hp'] == 'default']['mean_test_score'].to_list()

    for model_name in df_all['model_name'].unique():

        correct_model = df_all['model_name'] == model_name
        correct_task = df_all['hp'] == 'default'
        correct_benchmark = df_all['benchmark'] == sweep.benchmark + '_' + sweep.dataset_size

        default_runs = df_all.loc[correct_model & correct_task & correct_benchmark]
        default_runs.sort_values(by='data__keyword', inplace=True, ascending=True)

        df_new.loc[model_name] = default_runs['mean_test_score'].tolist()
    
    df_new.rename(columns=dict(zip(sweep.task_ids, sweep.dataset_names)), inplace=True)
    df_new.to_csv(sweep.path / DEFAULT_RESULTS_FILE_NAME, mode='w', index=True, header=True)


def make_random_sweep_plots(sweep: SweepConfig):

    results_path = sweep.path / RESULTS_FILE_NAME
    
    df = pd.read_csv(results_path)
    
    df['benchmark'] = sweep.benchmark + '_' + sweep.dataset_size

    df['data__openmlid'] = df['data__keyword']
    df['data__keyword'] = df['data__openmlid'].map(dict(zip(sweep.task_ids, sweep.dataset_names)))
    df['model_name'] = sweep.plot_name

    df.to_csv(sweep.path / RESULTS_MODIFIED_FILE_NAME, mode='w', index=False, header=True)

    script_name = f"bench_script_{sweep.benchmark}"
    script_path = 'analyses/' + script_name + '.R'
    results_csv_path = str(sweep.path)
    subprocess.run(['Rscript', script_path, results_csv_path])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--delay_in_seconds', type=int, default=10, help='Delay between checking if sweep is finished')

    args = parser.parse_args()

    monitor_and_make_plots(args.output_dir, args.delay_in_seconds)

