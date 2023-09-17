from __future__ import annotations

import pandas as pd

from tabularbench.core.enums import SearchType
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.paths_and_filenames import (
    RESULTS_FILE_NAME, DEFAULT_RESULTS_FILE_NAME
)


def make_default_results(sweep: SweepConfig):

    df_cur = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)
    df_cur['model'] = sweep.model_plot_name

    df_bench = get_benchmark_csv_reformatted()
    df = pd.concat([df_bench, df_cur], ignore_index=True)

    df.sort_values(by=['model', 'openml_dataset_name'], inplace=True)

    benchmark_plot_names = sweep.plotting.benchmark_models

    assert sweep.model_plot_name not in benchmark_plot_names, f"Don't use plot name {sweep.model_plot_name}, the benchmark already has a model with that name"

    index = benchmark_plot_names + [sweep.model_plot_name]
    df_new = pd.DataFrame(columns=sorted(sweep.openml_dataset_names), index=index, dtype=float)

    for model_name in index:

        correct_dataset = df['openml_dataset_name'].isin(sweep.openml_dataset_names)
        correct_model = df['model'] == model_name
        correct_search_type = df['search_type'] == SearchType.DEFAULT.name 
        correct_dataset_size = df['dataset_size'] == sweep.dataset_size.name
        correct_feature_type = df['feature_type'] == sweep.feature_type.name
        correct_task = df['task'] == sweep.task.name

        correct_all = correct_model & correct_search_type & correct_dataset_size & correct_feature_type & correct_task & correct_dataset

        default_runs = df.loc[correct_all]

        df_new.loc[model_name] = default_runs['score_test_mean'].tolist()
    
    df_new = df_new.applymap("{:.4f}".format)
    df_new.to_csv(sweep.sweep_dir / DEFAULT_RESULTS_FILE_NAME, mode='w', index=True, header=True)