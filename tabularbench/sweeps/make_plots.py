from __future__ import annotations

from pathlib import Path

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.sweeps.make_plots_tabzilla import plot_results_tabzilla
from tabularbench.sweeps.make_plots_whytrees import plot_results_whytrees
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

    path = Path('outputs_done/ablation/foundation_basesize_8/test-numerical-classification/')
    results_sweep = ResultsSweep.load(path / RESULTS_FILE_NAME)
    cfg = ConfigBenchmarkSweep.load(path / CONFIG_BENCHMARK_SWEEP_FILE_NAME)
    cfg.output_dir = path

    plot_results(cfg, results_sweep)


