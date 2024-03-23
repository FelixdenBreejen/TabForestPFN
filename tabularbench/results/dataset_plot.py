import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.results.dataset_manipulations import (add_model_plot_names, add_placeholder_as_model_name_dim,
                                                        only_use_models_and_datasets_specified_in_cfg)
from tabularbench.results.dataset_plot_combined import make_combined_dataset_plot, make_combined_dataset_plot_data
from tabularbench.results.dataset_plot_separate import make_separate_dataset_plot_data, make_separate_dataset_plots
from tabularbench.results.random_sequence import create_random_sequences_from_dataset
from tabularbench.results.reformat_results_get import get_reformatted_results_whytrees
from tabularbench.results.results_sweep import ResultsSweep


def make_dataset_plots(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    ds_whytrees = get_reformatted_results_whytrees()
    ds = combine_and_process_run_results_with_benchmark(cfg, results_sweep.ds, ds_whytrees)
    sequences_all = create_random_sequences_from_dataset(cfg, ds)

    plot_data_combined = make_combined_dataset_plot_data(cfg, sequences_all)
    np.save(cfg.output_dir / "dataset_plot_combined.npy", plot_data_combined)
    fig_combined = make_combined_dataset_plot(cfg, plot_data_combined)
    fig_combined.savefig(cfg.output_dir / "dataset_plot_combined.png")
    plt.close(fig_combined)

    plot_data_separate = make_separate_dataset_plot_data(cfg, sequences_all)
    np.save(cfg.output_dir / "dataset_plot_separate.npy", plot_data_separate)
    fig_separate = make_separate_dataset_plots(cfg, plot_data_separate)
    fig_separate.savefig(cfg.output_dir / "dataset_plot_separate.png")
    plt.close(fig_separate)


def combine_and_process_run_results_with_benchmark(cfg: ConfigBenchmarkSweep, ds_results: xr.Dataset, ds_whytrees: xr.Dataset) -> xr.Dataset:
    
    ds_whytrees = only_use_models_and_datasets_specified_in_cfg(cfg, ds_whytrees)
    ds_whytrees = add_model_plot_names(ds_whytrees)
    
    ds_results = add_placeholder_as_model_name_dim(ds_results, cfg.model_plot_name)

    ds = xr.merge([ds_whytrees, ds_results])

    return ds

