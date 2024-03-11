from __future__ import annotations

import pandas as pd
import xarray as xr

from tabularbench.core.enums import DataSplit, ModelName, SearchType
from tabularbench.results.reformat_whytrees_benchmark import get_whytrees_benchmark_reformatted
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.results.scores_min_max import get_combined_normalized_scores
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME


def make_default_results(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names]

    ds_whytrees = get_whytrees_benchmark_reformatted()
    ds_whytrees = ds_whytrees.sel(openml_dataset_id=cfg.openml_dataset_ids_to_use, model_name=benchmark_model_names)
    ds_whytrees = ds_whytrees.where(ds_whytrees['search_type'] == SearchType.DEFAULT.name, drop=True)

    model_plot_names = [ ModelName[x].value for x in ds_whytrees['model_name'].values ]
    ds_whytrees['model_name'] = xr.DataArray(model_plot_names, coords=dict(model_name=ds_whytrees.coords['model_name']))

    ds = results_sweep.ds
    ds = ds.where(ds['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds = ds.where(ds['seed'] == cfg.seed, drop=True) # when using multiple default runs, the seed changes
    

    df_run_results = df_run_results[ df_run_results['search_type'] == SearchType.DEFAULT.name ]
    df_run_results = df_run_results[ df_run_results['seed'] == cfg.seed ] # when using multiple default runs, the seed changes
    df_run_results['model_plot_name'] = cfg.model_plot_name
    df_run_results.sort_values(by=['openml_dataset_id'], inplace=True)

    df = pd.concat([df_bench, df_run_results], ignore_index=True)
    
    df['openml_dataset_name'] = df.apply(lambda row: row['openml_dataset_name'][:8] + '...' if len(row['openml_dataset_name']) > 11 else row['openml_dataset_name'], axis=1)

    make_df_results(cfg, df, DataSplit.VALID)
    make_df_results(cfg, df, DataSplit.TEST)


def make_df_results(cfg: ConfigBenchmarkSweep, df: pd.DataFrame, data_split: DataSplit) -> pd.DataFrame:

    match data_split:
        case DataSplit.VALID:
            score_name = 'score_val_mean'
            file_name = DEFAULT_RESULTS_VAL_FILE_NAME
        case DataSplit.TEST:
            score_name = 'score_test_mean'
            file_name = DEFAULT_RESULTS_TEST_FILE_NAME

    normalized_scores = calculate_normalized_scores(cfg, df, data_split)

    df_results = df.pivot(index=['model', 'model_plot_name'], columns=['openml_dataset_id', 'openml_dataset_name'], values=score_name)
    df_results['Normalized Score'] = df_results.apply(lambda row: normalized_scores[row.name[0]], axis=1)
    df_results.index = df_results.index.droplevel(0)
    df_results = df_results.map(lambda x: f"{x:.4f}")

    df_results.to_csv(cfg.output_dir / file_name, mode='w', index=True, header=True)





def calculate_normalized_scores(cfg: ConfigBenchmarkSweep, df: pd.DataFrame, data_split: DataSplit) -> dict[str, float]:

    match data_split:
        case DataSplit.VALID:
            score_name = 'score_val_mean'
        case DataSplit.TEST:
            score_name = 'score_test_mean'

    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names] + [cfg.model_name.name]
    
    normalized_scores = {}
    for model in benchmark_model_names:
        df_model = df[ df['model'] == model ]
        openml_dataset_ids = df_model['openml_dataset_id'].values.tolist()
        scores = df_model[score_name].values.tolist()
        normalized_score = get_combined_normalized_scores(cfg, openml_dataset_ids, data_split, scores)
        normalized_scores[model] = normalized_score

    return normalized_scores