import math
from matplotlib import pyplot as plt
import numpy as np

from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep


def make_separate_dataset_plot_data(cfg: ConfigBenchmarkSweep, sequences_all: np.ndarray) -> np.ndarray:
    
    n_dataset_ids = len(cfg.openml_dataset_ids_to_use)
    n_models = sequences_all.shape[0]

    plot_data = np.empty((3, n_models, n_dataset_ids, cfg.config_plotting.n_runs))
    
    for dataset_i in range(n_dataset_ids):
        for model_i in range(n_models):

            sequences = sequences_all[model_i, dataset_i, :, :]
            sequence_mean = np.mean(sequences, axis=0)
            sequence_lower_bound = np.quantile(sequences, q=1-cfg.config_plotting.confidence_bound, axis=0)
            sequence_upper_bound = np.quantile(sequences, q=cfg.config_plotting.confidence_bound, axis=0)

            plot_data[0, model_i, dataset_i, :] = sequence_mean
            plot_data[1, model_i, dataset_i, :] = sequence_lower_bound
            plot_data[2, model_i, dataset_i, :] = sequence_upper_bound

    return plot_data


def make_separate_dataset_plots(cfg: ConfigBenchmarkSweep, plot_data: np.ndarray) -> plt.Figure:

    models = cfg.config_plotting.benchmark_model_names + [cfg.model_plot_name]
    n_dataset_ids = len(cfg.openml_dataset_ids_to_use)

    num_horizontal_subplots = math.ceil(math.sqrt(n_dataset_ids))
    fig, axs = plt.subplots(nrows=num_horizontal_subplots, ncols=num_horizontal_subplots, figsize=(25, 25), )
    axs = axs.flatten()

    #remove the last few subplots that we don't need
    for i in range(n_dataset_ids, len(axs)):
        fig.delaxes(axs[i])
    
    for dataset_i, (openml_dataset_id, ax) in enumerate(zip(cfg.openml_dataset_ids_to_use, axs)):
        for model_i, model in enumerate(models):

            sequence_mean = plot_data[0, model_i, dataset_i, :]
            sequence_lower_bound = plot_data[1, model_i, dataset_i, :]
            sequence_upper_bound = plot_data[2, model_i, dataset_i, :]

            epochs = np.arange(len(sequence_mean)) + cfg.config_plotting.plot_default_value

            ax.plot(epochs, sequence_mean, label=model, linewidth=6)
            ax.fill_between(
                x=epochs, 
                y1=sequence_lower_bound, 
                y2=sequence_upper_bound, 
                alpha=0.2
            )

        dataset_name = cfg.benchmark.openml_dataset_names[cfg.benchmark.openml_dataset_ids.index(openml_dataset_id)]
        ax.set_title(dataset_name + f"({openml_dataset_id})", fontsize=30)
        ax.title.set_size(30)
        ax.set_xlabel("Number of runs")
        ax.xaxis.label.set_size(30)
        ax.set_ylabel("Test score")
        ax.yaxis.label.set_size(30)
        ax.tick_params(axis='both', which='major', labelsize=15)

        min_y = np.min(plot_data[1, :, dataset_i, :])
        max_y = np.max(plot_data[2, :, dataset_i, :])
        spread = max_y - min_y
        min_y = min_y - 0.1 * spread
        max_y = max_y + 0.1 * spread

        ax.set_ylim([min_y, max_y])
        ax.set_xscale('log')
        ax.set_xlim([1, cfg.config_plotting.n_runs])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))


    fig.suptitle(f"Test Score \n for all datasets of benchmark {cfg.benchmark.name}", fontsize=50)
    fig.tight_layout(pad=2.0, rect=[0.05, 0.12, 0.90, 1.00])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=40, handlelength=3)

    return fig


