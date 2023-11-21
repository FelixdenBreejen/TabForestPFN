from __future__ import annotations
import argparse
import time

import pandas as pd


from tabularbench.core.enums import SearchType
from tabularbench.results.random_sweep_plots import make_random_sweep_plots
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.datasets import get_unfinished_dataset_ids
from tabularbench.sweeps.paths_and_filenames import (
    DEFAULT_RESULTS_FILE_NAME,
    RESULTS_FILE_NAME
)
from tabularbench.sweeps.get_logger import get_logger
from tabularbench.sweeps.writer import Writer
from tabularbench.results.default_results import make_default_results
from tabularbench.results.hyperparam_plots import make_hyperparam_plots


def plot_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    if len(df_run_results) == 0:
        # no results yet to plot
        return

    cfg.logger.info(f"Start making plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")

    if sweep_default_finished(cfg, df_run_results) and default_results_not_yet_made(cfg):
        cfg.logger.info(f"Start making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")
        make_default_results(cfg, df_run_results)
        cfg.logger.info(f"Finished making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")

    
    if cfg.search_type == SearchType.RANDOM:
        cfg.logger.info(f"Start making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_hyperparam_plots(cfg, df_run_results)
        cfg.logger.info(f"Finished making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    

    cfg.logger.info(f"Start making sweep plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    # make_sweep_plots(cfg, df_run_results)
    cfg.logger.info(f"Finished making sweep plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    
    cfg.logger.info(f"Finished making plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")




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


def sweep_default_finished(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    df = df_run_results
    df = df[ df['search_type'] == SearchType.DEFAULT.name ]
    df = df[ df['seed'] == cfg.seed ]    # when using multiple default runs, the seed changes

    for dataset_id in cfg.openml_dataset_ids_to_use:

        df_id = df[ df['openml_dataset_id'] == dataset_id ]
        if len(df_id) == 0:
            return False

    return True


def default_results_not_yet_made(cfg: ConfigBenchmarkSweep) -> bool:
    return not (cfg.output_dir / DEFAULT_RESULTS_FILE_NAME).exists()



def sweep_random_finished(sweep) -> bool:
    
    unfinished_tasks = get_unfinished_dataset_ids(sweep.openml_dataset_ids, sweep.sweep_dir / RESULTS_FILE_NAME, runs_per_dataset=sweep.runs_per_dataset)
    return len(unfinished_tasks) == 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--delay_in_seconds', type=int, default=10, help='Delay between checking if sweep is finished')

    args = parser.parse_args()

    monitor_and_make_plots(args.output_dir, args.delay_in_seconds)

