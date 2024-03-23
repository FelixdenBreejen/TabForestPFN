from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.core.enums import DataSplit, ModelName, SearchType
from tabularbench.results.dataset_manipulations import (add_model_plot_names, add_placeholder_as_model_name_dim,
                                                        select_only_the_first_default_run_of_every_model_and_dataset)
from tabularbench.results.reformat_results_get import get_reformatted_results_tabzilla
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.results.scores_min_max import get_combined_normalized_scores
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME


def make_default_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    ds_whytrees = process_tabzilla_benchmark_results(cfg)
    ds_sweep = process_sweep_results(cfg, results_sweep)

    ds = xr.merge([ds_whytrees, ds_sweep])

    make_df_results(cfg, ds, DataSplit.VALID)
    make_df_results(cfg, ds, DataSplit.TEST)


def process_tabzilla_benchmark_results(cfg: ConfigBenchmarkSweep) -> xr.Dataset:

    benchmark_model_names = [model_name.name for model_name in cfg.plotting.tabzilla.benchmark_model_names]

    ds_whytrees = get_reformatted_results_tabzilla()
    ds_whytrees = ds_whytrees.sel(openml_dataset_id=cfg.openml_dataset_ids_to_use, model_name=benchmark_model_names)
    vars_with_run_id = ['search_type', 'score', 'runs_actual']
    ds_whytrees[vars_with_run_id] = ds_whytrees[vars_with_run_id].where(ds_whytrees['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds_whytrees = ds_whytrees.sum(dim='run_id', keep_attrs=True)

    ds_whytrees = add_model_plot_names(ds_whytrees)

    return ds_whytrees


def process_sweep_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> xr.Dataset:

    ds = results_sweep.ds.copy()
    ds = add_placeholder_as_model_name_dim(ds, cfg.model_plot_name)
    ds = select_only_the_first_default_run_of_every_model_and_dataset(cfg, ds)
    
    return ds



def make_df_results(cfg: ConfigBenchmarkSweep, ds: xr.Dataset, data_split: DataSplit) -> pd.DataFrame:

    normalized_scores = calculate_normalized_scores(cfg, ds, data_split)

    ds = ds.sel(data_split=data_split.name)
    score = ds['score'].sum(dim='cv_split', keep_attrs=True) / ds['cv_splits_actual']

    score_values = score.values
    normalized_score_values = [normalized_scores[ModelName[model_name]] for model_name in score.coords['model_name'].values]
    normalized_score_array = np.array(normalized_score_values)[:, None]
    score_values = np.concatenate([score_values, normalized_score_array], axis=1)

    score = xr.DataArray(
        data=score_values, 
        dims=['model_plot_name', 'openml_dataset_id'], 
        coords = {
            'model_plot_name': ds['model_plot_name'].values,
            'openml_dataset_id': ds.coords['openml_dataset_id'].values.tolist() + ['Aggregate']
        }
    )

    benchmark_model_names =  cfg.plotting.get_benchmark_model_names(cfg.benchmark.origin)
    model_plot_names = [model_name.value for model_name in benchmark_model_names] + [cfg.model_plot_name]
    score = score.reindex(model_plot_name = model_plot_names)
    df = score.to_pandas()
    df = df.round(4)
    df.to_csv(cfg.output_dir / get_results_file_name(data_split), mode='w', index=True, header=True)


def get_results_file_name(data_split: DataSplit):

    match data_split:
        case DataSplit.VALID:
            return DEFAULT_RESULTS_VAL_FILE_NAME
        case DataSplit.TEST:
            return DEFAULT_RESULTS_TEST_FILE_NAME


def calculate_normalized_scores(cfg: ConfigBenchmarkSweep, ds: xr.Dataset, data_split: DataSplit) -> dict[ModelName, float]:

    benchmark_model_names = cfg.plotting.get_benchmark_model_names(cfg.benchmark.origin) + [ModelName.PLACEHOLDER]
    
    normalized_scores = {}
    for model_name in benchmark_model_names:

        scores = ds['score'].sel(model_name=model_name.name, data_split=data_split.name)
        scores = scores.sum(dim='cv_split') / ds['cv_splits_actual']
        scores = scores.values.tolist()

        openml_dataset_ids = ds.coords['openml_dataset_id'].values.tolist()

        normalized_score = get_combined_normalized_scores(cfg, openml_dataset_ids, data_split, scores)
        normalized_scores[model_name] = normalized_score

    return normalized_scores