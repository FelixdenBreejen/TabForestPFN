from __future__ import annotations

from pathlib import Path

from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.sweeps.make_plots_tabzilla import plot_results_tabzilla
from tabularbench.sweeps.make_plots_whytrees import plot_results_whytrees
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import CONFIG_BENCHMARK_SWEEP_FILE_NAME, RESULTS_FILE_NAME


def plot_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    if results_sweep.ds.sizes['run_id'] == 0:
        # no results yet to plot
        return
    
    match cfg.benchmark.origin:
        case BenchmarkOrigin.TABZILLA:
            plot_results_tabzilla(cfg, results_sweep)
        case BenchmarkOrigin.WHYTREES:
            plot_results_whytrees(cfg, results_sweep)


if __name__ == "__main__":

    path = Path('outputs/2024-03-19/14-25-45/foundation-default-debug_tabzilla')
    results_sweep = ResultsSweep.load(path / RESULTS_FILE_NAME)
    cfg = ConfigBenchmarkSweep.load(path / CONFIG_BENCHMARK_SWEEP_FILE_NAME)

    plot_results(cfg, results_sweep)


