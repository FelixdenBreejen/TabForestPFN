from __future__ import annotations

import pandas as pd

from tabularbench.core.enums import DataSplit, ModelName, SearchType
from tabularbench.results.reformat_benchmark import \
    get_benchmark_csv_reformatted
from tabularbench.results.scores_min_max import get_combined_normalized_scores
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import (
    DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME)


def make_default_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names]

    df_bench = get_benchmark_csv_reformatted()
    df_bench = df_bench[ df_bench['openml_dataset_id'].isin(cfg.openml_dataset_ids_to_use) ]
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench = df_bench[ df_bench['search_type'] == SearchType.DEFAULT.name ]
    df_bench['model_plot_name'] = df_bench.apply(lambda row: ModelName[row['model']].value, axis=1)
    df_bench.sort_values(by=['model', 'openml_dataset_id'], inplace=True)

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