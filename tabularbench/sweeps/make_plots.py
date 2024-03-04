from __future__ import annotations

import pandas as pd
from loguru import logger

from tabularbench.core.enums import SearchType
from tabularbench.results.dataset_plot import make_dataset_plots
from tabularbench.results.default_results import make_default_results
from tabularbench.results.hyperparam_plots import make_hyperparam_plots
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import \
    DEFAULT_RESULTS_TEST_FILE_NAME


def plot_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    if len(df_run_results) == 0:
        # no results yet to plot
        return

    if sweep_default_finished(cfg, df_run_results) and default_results_not_yet_made(cfg):
        logger.info(f"Start making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")
        make_default_results(cfg, df_run_results)
        logger.info(f"Finished making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")


    if cfg.search_type == SearchType.RANDOM:
        logger.info(f"Start making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_hyperparam_plots(cfg, df_run_results)
        logger.info(f"Finished making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    
    
    if sweep_default_finished(cfg, df_run_results):
        logger.info(f"Start making dataset plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_dataset_plots(cfg, df_run_results)
        logger.info(f"Finished making dataset plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")



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
    return not (cfg.output_dir / DEFAULT_RESULTS_TEST_FILE_NAME).exists()


