from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import time

import pandas as pd
import numpy as np

from tabularbench.configs.all_model_configs import total_config
from tabularbench.sweeps.sweep_config import SweepConfig, load_sweep_configs_from_file
from tabularbench.sweeps.datasets import get_unfinished_task_ids
from tabularbench.sweeps.paths_and_filenames import (
    SWEEP_FILE_NAME, RESULTS_FILE_NAME, RESULTS_MODIFIED_FILE_NAME, 
    PATH_TO_ALL_BENCH_CSV, DEFAULT_RESULTS_FILE_NAME
)


def monitor_and_make_plots(process_id: int, output_dir: str, delay_in_seconds: int = 10):

    sweep_csv = pd.read_csv(Path(output_dir) / SWEEP_FILE_NAME)
    sweeps = load_sweep_configs_from_file(Path(output_dir) / SWEEP_FILE_NAME)

    for sweep in sweeps:

        while True:
            if sweep_default_finished(sweep):
                make_default_results(sweep)
                break
            
            time.sleep(delay_in_seconds)

        while True:
                
            make_results_csv_modified_for_plotting(sweep)
            make_random_sweep_plots(sweep)

            if sweep.random_search:
                make_hyperparam_plots(sweep)
            
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

    benchmark_plot_names = df_all['model_name'].unique().tolist()
    # Not all benchmarks have this model, and also it's not a very important model.
    benchmark_plot_names.remove('HistGradientBoostingTree')

    assert sweep.plot_name not in benchmark_plot_names, f"Don't use plot name {sweep.plot_name}, the benchmark already has a model with that name"

    index = benchmark_plot_names + [sweep.plot_name]
    df_new = pd.DataFrame(columns=df['data__keyword'].unique().tolist(), index=index)

    df_new.loc[sweep.plot_name] = df[df['hp'] == 'default']['mean_test_score'].to_list()

    for model_name in benchmark_plot_names:

        correct_model = df_all['model_name'] == model_name
        correct_task = df_all['hp'] == 'default'
        correct_benchmark = df_all['benchmark'] == sweep.benchmark + '_' + sweep.dataset_size

        default_runs = df_all.loc[correct_model & correct_task & correct_benchmark]
        default_runs.sort_values(by='data__keyword', inplace=True, ascending=True)

        df_new.loc[model_name] = default_runs['mean_test_score'].tolist()
    
    df_new.rename(columns=dict(zip(sweep.task_ids, sweep.dataset_names)), inplace=True)
    df_new.to_csv(sweep.path / DEFAULT_RESULTS_FILE_NAME, mode='w', index=True, header=True)


def make_results_csv_modified_for_plotting(sweep: SweepConfig):

    results_path = sweep.path / RESULTS_FILE_NAME
    
    df = pd.read_csv(results_path)
    
    df['benchmark'] = sweep.benchmark + '_' + sweep.dataset_size

    df['data__openmlid'] = df['data__keyword']
    df['data__keyword'] = df['data__openmlid'].map(dict(zip(sweep.task_ids, sweep.dataset_names)))
    df['model_name'] = sweep.plot_name

    if not sweep.random_search:
        # for default sweep, we want a straight line, so we fake the results by duplicating them
        df_random_fake = df.copy()
        df_random_fake['hp'] = 'random'
        df = pd.concat([df] + [df_random_fake]*999)


    df.to_csv(sweep.path / RESULTS_MODIFIED_FILE_NAME, mode='w', index=False, header=True)


def make_random_sweep_plots(sweep: SweepConfig):

    script_name = f"bench_script_{sweep.benchmark}"
    script_path = 'analyses/' + script_name + '.R'
    results_csv_path = str(sweep.path)
    subprocess.run(['Rscript', script_path, results_csv_path])


def make_hyperparam_plots(sweep: SweepConfig):
    
    df = pd.read_csv(sweep.path / RESULTS_MODIFIED_FILE_NAME)
    config = total_config[sweep.model][sweep.task]

    for dataset_name in sweep.dataset_names:
        for random_var in config['random'].keys():
                
            this_dataset = df['data__keyword'] == dataset_name
            fig = None

            if 'min' in config['random'][random_var]:
                is_log = 'log' in config['random'][random_var]['distribution']
                fig = df[this_dataset].plot(kind='scatter', x=random_var, y='mean_test_score', logx=is_log).get_figure()

            elif 'values' in config['random'][random_var]:
                fig = df[this_dataset].boxplot(column='mean_test_score', by=random_var).get_figure()

            if fig is not None:
                png_path = sweep.path / 'hyperparam_plots' / f'{dataset_name}_{random_var}.png'
                png_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(png_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--delay_in_seconds', type=int, default=10, help='Delay between checking if sweep is finished')

    args = parser.parse_args()

    monitor_and_make_plots(args.output_dir, args.delay_in_seconds)

