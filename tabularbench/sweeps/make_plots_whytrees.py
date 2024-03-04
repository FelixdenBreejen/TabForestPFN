from __future__ import annotations

import pandas as pd
from loguru import logger

from tabularbench.core.enums import SearchType
from tabularbench.results.dataset_plot import make_dataset_plots
from tabularbench.results.default_results import make_default_results
from tabularbench.results.hyperparam_plots import make_hyperparam_plots
from tabularbench.sweeps.make_plots_utils import (default_results_not_yet_made,
                                                  sweep_default_finished)
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def plot_results_whytrees(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

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

