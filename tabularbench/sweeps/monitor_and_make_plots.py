from __future__ import annotations
import argparse
import subprocess
import time

import pandas as pd

from tabularbench.configs.all_model_configs import total_config
from tabularbench.core.enums import SearchType
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.sweep_config import SweepConfig, create_sweep_config_list_from_main_config
from tabularbench.sweeps.datasets import get_unfinished_dataset_ids
from tabularbench.sweeps.paths_and_filenames import (
    RESULTS_FILE_NAME, RESULTS_MODIFIED_FILE_NAME, 
    DEFAULT_RESULTS_FILE_NAME
)
from tabularbench.sweeps.sweep_start import get_config, get_logger, set_seed
from tabularbench.sweeps.writer import Writer


def monitor_and_make_plots(output_dir: str, writer: Writer, delay_in_seconds: int = 10):

    cfg = get_config(output_dir)
    cfg.seed = 0

    logger = get_logger(cfg, 'monitor_and_make_plots.log')

    sweep_configs = create_sweep_config_list_from_main_config(cfg, writer, logger)
    
    logger.info(f"Found {len(sweep_configs)} sweeps to monitor")
    logger.info(f"Start monitoring all sweeps")

    for sweep in sweep_configs:

        set_seed(sweep.seed)
        logger.info(f"Start monitoring sweep {str(sweep)}")

        while True:
            if sweep_default_finished(sweep):
                logger.info(f"Start making default results for sweep {str(sweep)}")
                make_default_results(sweep)
                logger.info(f"Finished making default results for sweep {str(sweep)}")
                break
            
            time.sleep(delay_in_seconds)

        
        while True:
            
            logger.info(f"Start making result plots for sweep {str(sweep)}")
            make_random_sweep_plots(sweep)
            logger.info(f"Finished making result plots for sweep {str(sweep)}")

            if sweep.search_type == SearchType.RANDOM:
                logger.info(f"Start making hyperparam plots for sweep {str(sweep)}")
                make_hyperparam_plots(sweep)
                logger.info(f"Finished making hyperparam plots for sweep {str(sweep)}")
            
            if sweep_random_finished(sweep) or sweep.search_type == SearchType.DEFAULT:
                break

            time.sleep(delay_in_seconds)

        logger.info(f"Finished monitoring sweep {str(sweep)}")

    logger.info(f"Finished monitoring all sweeps")


def sweep_default_finished(sweep: SweepConfig) -> bool:
    
    # default sweep always finishes before random sweep starts, so we just check if every dataset has one run
    unfinished_tasks = get_unfinished_dataset_ids(sweep.openml_dataset_ids, sweep.sweep_dir / RESULTS_FILE_NAME, runs_per_dataset=1)
    return len(unfinished_tasks) == 0


def sweep_random_finished(sweep: SweepConfig) -> bool:

    if not sweep.random_search:
        return True
    
    unfinished_tasks = get_unfinished_dataset_ids(sweep.openml_dataset_ids, sweep.sweep_dir / RESULTS_FILE_NAME, runs_per_dataset=sweep.runs_per_dataset)
    return len(unfinished_tasks) == 0


def make_default_results(sweep: SweepConfig):

    df_cur = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)
    df_cur['model'] = sweep.model_plot_name

    df_bench = get_benchmark_csv_reformatted()
    df = pd.concat([df_bench, df_cur], ignore_index=True)

    df.sort_values(by=['model', 'openml_dataset_name'], inplace=True)

    benchmark_plot_names = df_bench['model'].unique().tolist()
    # Not all benchmarks have this model, and also it's not a very important model.
    benchmark_plot_names.remove('HistGradientBoostingTree')

    assert sweep.model_plot_name not in benchmark_plot_names, f"Don't use plot name {sweep.model_plot_name}, the benchmark already has a model with that name"

    index = benchmark_plot_names + [sweep.model_plot_name]
    df_new = pd.DataFrame(columns=df_cur['openml_dataset_name'].unique().tolist(), index=index, dtype=float)

    for model_name in index:

        correct_model = df['model'] == model_name
        correct_search_type = df['search_type'] == SearchType.DEFAULT.name 
        correct_dataset_size = df['dataset_size'] == sweep.dataset_size.name
        correct_feature_type = df['feature_type'] == sweep.feature_type.name
        correct_task = df['task'] == sweep.task.name

        correct_all = correct_model & correct_search_type & correct_dataset_size & correct_feature_type & correct_task

        default_runs = df.loc[correct_all]

        df_new.loc[model_name] = default_runs['score_test_mean'].tolist()
    
    df_new = df_new.applymap("{:.4f}".format)
    df_new.to_csv(sweep.sweep_dir / DEFAULT_RESULTS_FILE_NAME, mode='w', index=True, header=True)



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

