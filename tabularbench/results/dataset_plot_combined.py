from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from tabularbench.core.enums import ModelName, SearchType
from tabularbench.results.random_sequence import create_random_sequences
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.results.scores_min_max import scores_min_max
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep


def make_combined_dataset_plot(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names]

    df_bench = get_benchmark_csv_reformatted()
    df_bench = df_bench[ df_bench['openml_dataset_id'].isin(cfg.openml_dataset_ids_to_use) ]
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench['model_plot_name'] = df_bench.apply(lambda row: ModelName[row['model']].value, axis=1)
    df_bench.sort_values(by=['model', 'openml_dataset_id'], inplace=True)

    df_run_results['model_plot_name'] = cfg.model_plot_name
    df_run_results.sort_values(by=['openml_dataset_id'], inplace=True)

    df = pd.concat([df_bench, df_run_results], ignore_index=True)

    fig, ax, plot_data = make_combined_dataset_plot_(cfg, df)

    fig.savefig(cfg.output_dir / "combined_dataset_plot.png")
    np.save(cfg.output_dir / "plot_data.npy", plot_data)


def make_combined_dataset_plot_(cfg: ConfigBenchmarkSweep, df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes, np.ndarray]:

    models = df['model'].unique().tolist()

    sequences_all = np.zeros((len(models), len(cfg.openml_dataset_ids_to_use), cfg.config_plotting.n_random_shuffles, cfg.config_plotting.n_runs))

    for dataset_i, openml_dataset_id in enumerate(sorted(cfg.openml_dataset_ids_to_use)):

        df_dataset = df[ df['openml_dataset_id'] == openml_dataset_id ]
        score_min, score_max = scores_min_max(cfg, openml_dataset_id)

        for model_i, model in enumerate(models):

            df_model = df_dataset[ df_dataset['model'] == model ]

            df_model_default = df_model[ df_model['search_type'] == SearchType.DEFAULT.name ]
            df_model_default_seed_0 = df_model_default[ df_model_default['seed'] == cfg.seed ]

            if len(df_model_default) == 1:
                # If there is one default value, we use that
                default_value_val = df_model_default['score_val_mean'].item()
                default_value_test = df_model_default['score_test_mean'].item()
            elif len(df_model_default_seed_0) == 1:
                # If there are multiple default values, we use the one with seed 0
                default_value_val = df_model_default_seed_0['score_val_mean'].item()
                default_value_test = df_model_default_seed_0['score_test_mean'].item()
            elif len(df_model_default) == 0:
                cfg.logger.warning(f"No default value found for model {model} on dataset {openml_dataset_id}. We will assume 0.")
                default_value_val = 0
                default_value_test = 0
            else:
                raise ValueError(f"More than one default value found for model {model} on dataset {openml_dataset_id}")
            
            df_model_random = df_model[ df_model['search_type'] == SearchType.RANDOM.name ]
            random_values_val = df_model_random['score_val_mean'].values
            random_values_test = df_model_random['score_test_mean'].values

            sequences = create_random_sequences(
                default_value_val = default_value_val, 
                default_value_test = default_value_test,
                random_values_val = random_values_val,
                random_values_test = random_values_test,
                sequence_length = cfg.config_plotting.n_runs,
                n_shuffles = cfg.config_plotting.n_random_shuffles
            )
            sequences = sequences.clip(min=0)

            sequences_normalized = (sequences - score_min).clip(min=0) / (score_max - score_min)

            sequences_all[model_i, dataset_i, :, :] = sequences_normalized


    sequences_all = np.mean(sequences_all, axis=1)   # [models, sequence_length, n_shuffles]
    
    fig, ax = plt.subplots(figsize=(25, 25))

    plot_data = np.empty((3, len(models), cfg.config_plotting.n_runs))

    for model_i, model in enumerate(models):

        sequence_mean = np.mean(sequences_all[model_i, :, :], axis=0)
        sequence_lower_bound = np.quantile(sequences_all[model_i, :, :], q=1-cfg.config_plotting.confidence_bound, axis=0)
        sequence_upper_bound = np.quantile(sequences_all[model_i, :, :], q=cfg.config_plotting.confidence_bound, axis=0)

        plot_data[0, model_i, :] = sequence_mean
        plot_data[1, model_i, :] = sequence_lower_bound
        plot_data[2, model_i, :] = sequence_upper_bound

        # color_and_linestyle_for_model_generator.send(None)
        # color, linestyle = color_and_linestyle_for_model_generator.send(model)

        ax.plot(sequence_mean, label=model, linewidth=12)
        ax.fill_between(
            x=np.arange(len(sequence_mean)), 
            y1=sequence_lower_bound, 
            y2=sequence_upper_bound, 
            alpha=0.2
        )


    ax.set_title(f"Averaged Normalized Test Score \n for all datasets of benchmark {cfg.benchmark.name}", fontsize=50)
    ax.set_xlabel("Number of runs", fontsize=50)
    ax.set_ylabel("Normalized Test score", fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=40)

    ax.set_xscale('log')
    ax.set_xlim([1, cfg.config_plotting.n_runs])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=40, handlelength=3)
    fig.tight_layout(pad=2.0, rect=[0, 0.12, 1, 0.98])

    return fig, ax, plot_data














