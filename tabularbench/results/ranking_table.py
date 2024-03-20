from __future__ import annotations

import pandas as pd
import xarray as xr

from tabularbench.core.enums import DataSplit
from tabularbench.results.dataset_manipulations import (add_model_plot_names, add_placeholder_as_model_name_dim,
                                                        average_out_the_cv_split, only_use_models_and_datasets_specified_in_cfg,
                                                        select_only_the_first_default_run_of_every_model_and_dataset,
                                                        take_run_with_best_validation_loss)
from tabularbench.results.reformat_results_get import get_reformatted_results
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.results.scores_min_max import normalize_scores
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import RANKING_TABLE_FILE_NAME


def make_ranking_table(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:
    pass

    ds_benchmark = process_benchmark_results(cfg)
    ds_sweep = process_sweep_results(cfg, results_sweep)

    ds = xr.merge([ds_benchmark, ds_sweep], combine_attrs='drop')

    make_ranking_table_(cfg, ds)


def process_benchmark_results(cfg: ConfigBenchmarkSweep) -> xr.Dataset:

    ds_benchmark = get_reformatted_results(cfg.benchmark.origin)
    ds_benchmark = only_use_models_and_datasets_specified_in_cfg(cfg, ds_benchmark)
    ds_benchmark = average_out_the_cv_split(ds_benchmark)
    ds_benchmark = take_run_with_best_validation_loss(ds_benchmark)
    ds_benchmark = add_model_plot_names(ds_benchmark)

    return ds_benchmark



def process_sweep_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> xr.Dataset:

    ds = results_sweep.ds.copy()
    ds = add_placeholder_as_model_name_dim(ds, cfg.model_plot_name)
    ds = select_only_the_first_default_run_of_every_model_and_dataset(cfg, ds)
    ds = average_out_the_cv_split(ds)
    
    return ds


def make_ranking_table_(cfg: ConfigBenchmarkSweep, ds: xr.Dataset) -> None:

    ds['normalized_accuracy'] = normalize_scores(cfg, ds['accuracy'])

    ds = ds.sel(data_split=DataSplit.TEST.name, )

    metrics = {}

    ranks = (1-ds['accuracy']).rank(dim='model_name')
    metrics['rank_min'] = ranks.min(dim='openml_dataset_id').values
    metrics['rank_max'] = ranks.max(dim='openml_dataset_id').values
    metrics['rank_mean'] = ranks.mean(dim='openml_dataset_id').round(1).values
    metrics['rank_median'] = ranks.median(dim='openml_dataset_id').values

    metrics['acc_mean'] = ds['normalized_accuracy'].mean(dim='openml_dataset_id').round(3).values
    metrics['acc_median'] = ds['normalized_accuracy'].median(dim='openml_dataset_id').round(3).values

    df = pd.DataFrame(metrics, index=ds['model_plot_name'].values)
    df.sort_values(by='rank_mean', inplace=True)
    df.to_csv(cfg.output_dir / RANKING_TABLE_FILE_NAME)
