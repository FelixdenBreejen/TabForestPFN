from __future__ import annotations

from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.sweeps.make_plots_tabzilla import plot_results_tabzilla
from tabularbench.sweeps.make_plots_whytrees import plot_results_whytrees
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def plot_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    if results_sweep.ds.sizes['run_id'] == 0:
        # no results yet to plot
        return
    
    match cfg.benchmark.origin:
        case BenchmarkOrigin.TABZILLA:
            plot_results_tabzilla(cfg, results_sweep)
        case BenchmarkOrigin.WHYTREES:
            plot_results_whytrees(cfg, results_sweep)


