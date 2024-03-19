from __future__ import annotations

import pandas as pd
import xarray as xr

from tabularbench.core.enums import DataSplit
from tabularbench.results.dataset_manipulations import (add_model_plot_names, add_placeholder_as_model_name_dim,
                                                        average_out_the_cv_split, change_data_var_names,
                                                        only_use_models_and_datasets_specified_in_cfg,
                                                        select_only_default_runs_and_average_over_them,
                                                        select_only_the_first_default_run_of_every_model_and_dataset)
from tabularbench.results.reformat_results_get import get_reformatted_results
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.results.scores_min_max import normalize_scores
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME


def make_default_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    ds_benchmark = process_benchmark_results(cfg)
    ds_sweep = process_sweep_results(cfg, results_sweep)

    ds = xr.merge([ds_benchmark, ds_sweep])

    make_df_results(cfg, ds, DataSplit.VALID)
    make_df_results(cfg, ds, DataSplit.TEST)


def process_benchmark_results(cfg: ConfigBenchmarkSweep) -> xr.Dataset:

    ds_benchmark = get_reformatted_results(cfg.benchmark.origin)
    ds_benchmark = only_use_models_and_datasets_specified_in_cfg(cfg, ds_benchmark)
    ds_benchmark = select_only_default_runs_and_average_over_them(ds_benchmark)
    ds_benchmark = add_model_plot_names(ds_benchmark)
    ds_benchmark = change_data_var_names(ds_benchmark)

    return ds_benchmark



def process_sweep_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> xr.Dataset:

    ds = results_sweep.ds.copy()
    ds = add_placeholder_as_model_name_dim(ds, cfg.model_plot_name)
    ds = select_only_the_first_default_run_of_every_model_and_dataset(cfg, ds)
    
    return ds



def make_df_results(cfg: ConfigBenchmarkSweep, ds: xr.Dataset, data_split: DataSplit) -> pd.DataFrame:

    ds = average_out_the_cv_split(ds)
    ds['normalized_accuracy'] = normalize_scores(cfg, ds['accuracy'])
    ds = ds.sel(data_split=data_split.name).reset_coords('data_split', drop=True)

    df = ds['accuracy'].to_pandas()
    normalized_accuracy = ds['normalized_accuracy'].mean(dim='openml_dataset_id').to_dataframe()

    df['aggregate'] = normalized_accuracy['normalized_accuracy']
    df = df.set_index(ds['model_plot_name'].values)
    df = df.round(4)

    df.to_csv(cfg.output_dir / get_results_file_name(data_split), mode='w', header=True)


def get_results_file_name(data_split: DataSplit):

    match data_split:
        case DataSplit.VALID:
            return DEFAULT_RESULTS_VAL_FILE_NAME
        case DataSplit.TEST:
            return DEFAULT_RESULTS_TEST_FILE_NAME
