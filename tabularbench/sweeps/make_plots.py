from __future__ import annotations

import pandas as pd

from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.sweeps.make_plots_tabzilla import plot_results_tabzilla
from tabularbench.sweeps.make_plots_whytrees import plot_results_whytrees
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def plot_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    if len(df_run_results) == 0:
        # no results yet to plot
        return
    
    match cfg.benchmark.origin:
        case BenchmarkOrigin.TABZILLA:
            plot_results_tabzilla(cfg, df_run_results)
        case BenchmarkOrigin.WHYTREES:
            plot_results_whytrees(cfg, df_run_results)


