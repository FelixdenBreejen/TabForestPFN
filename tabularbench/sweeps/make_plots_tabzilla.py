from __future__ import annotations

from loguru import logger

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.core.enums import SearchType
from tabularbench.results.default_results import make_default_results
from tabularbench.results.hyperparam_plots import make_hyperparam_plots
from tabularbench.results.ranking_table import make_ranking_table
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.sweeps.make_plots_utils import default_results_not_yet_made, sweep_default_finished


def plot_results_tabzilla(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    if sweep_default_finished(cfg, results_sweep) and default_results_not_yet_made(cfg):
        logger.info(f"Start making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")
        make_default_results(cfg, results_sweep)
        logger.info(f"Finished making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")


    if cfg.search_type == SearchType.RANDOM:
        logger.info(f"Start making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_hyperparam_plots(cfg, results_sweep)
        logger.info(f"Finished making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    
    
    if sweep_default_finished(cfg, results_sweep):
        logger.info(f"Start making ranking table for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_ranking_table(cfg, results_sweep)
        logger.info(f"Finished making ranking table for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")


