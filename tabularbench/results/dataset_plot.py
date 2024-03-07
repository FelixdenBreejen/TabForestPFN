import numpy as np
import pandas as pd

from tabularbench.core.enums import ModelName
from tabularbench.results.dataset_plot_combined import make_combined_dataset_plot, make_combined_dataset_plot_data
from tabularbench.results.dataset_plot_separate import make_separate_dataset_plot_data, make_separate_dataset_plots
from tabularbench.results.random_sequence import create_random_sequences_from_df
from tabularbench.results.reformat_whytrees_benchmark import get_whytrees_benchmark_reformatted
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def make_dataset_plots(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    df_bench = get_whytrees_benchmark_reformatted()
    df_combined = combine_and_process_run_results_with_benchmark(cfg, df_run_results, df_bench)
    sequences_all = create_random_sequences_from_df(cfg, df_combined)

    plot_data_combined = make_combined_dataset_plot_data(cfg, sequences_all)
    np.save(cfg.output_dir / "dataset_plot_combined.npy", plot_data_combined)
    fig_combined = make_combined_dataset_plot(cfg, plot_data_combined)
    fig_combined.savefig(cfg.output_dir / "dataset_plot_combined.png")

    plot_data_separate = make_separate_dataset_plot_data(cfg, sequences_all)
    np.save(cfg.output_dir / "dataset_plot_separate.npy", plot_data_separate)
    fig_separate = make_separate_dataset_plots(cfg, plot_data_separate)
    fig_separate.savefig(cfg.output_dir / "dataset_plot_separate.png")


def combine_and_process_run_results_with_benchmark(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    
    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names]

    df_bench = df_bench[ df_bench['openml_dataset_id'].isin(cfg.openml_dataset_ids_to_use) ]
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench['model_plot_name'] = df_bench.apply(lambda row: ModelName[row['model']].value, axis=1)
    df_bench.sort_values(by=['model', 'openml_dataset_id'], inplace=True)

    df_run_results['model'] = ModelName.PLACEHOLDER.name    # The model might be named the same as one of the benchmark models.
    df_run_results['model_plot_name'] = cfg.model_plot_name
    df_run_results.sort_values(by=['openml_dataset_id'], inplace=True)

    df = pd.concat([df_bench, df_run_results], ignore_index=True)

    return df

