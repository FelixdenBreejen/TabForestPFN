import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabularbench.core.enums import SearchType

from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.paths_and_filenames import RESULTS_FILE_NAME


def make_separate_dataset_plots(sweep):

    df_bench = get_benchmark_csv_reformatted()
    
    df_cur = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)
    df_cur['model'] = sweep.model_plot_name

    df = pd.concat([df_bench, df_cur], ignore_index=True)

    num_horizontal_subplots = math.ceil(math.sqrt(len(sweep.openml_dataset_names)))
    fig, axs = plt.subplots(nrows=num_horizontal_subplots, ncols=num_horizontal_subplots, figsize=(25, 25), )
    axs = axs.flatten()

    #remove the last few subplots that we don't need
    for i in range(len(sweep.openml_dataset_names), len(axs)):
        fig.delaxes(axs[i])

    for dataset_name, ax in zip(sorted(sweep.openml_dataset_names), axs):

        correct_dataset = df['openml_dataset_name'] == dataset_name
        correct_dataset_size = df['dataset_size'] == sweep.dataset_size.name
        correct_feature_type = df['feature_type'] == sweep.feature_type.name
        correct_task = df['task'] == sweep.task.name

        correct_all = correct_dataset & correct_dataset_size & correct_feature_type & correct_task
        df_correct = df.loc[correct_all]

        sequences_all = []        

        for model in sweep.plotting.benchmark_models + [sweep.model_plot_name]:

            correct_model = df_correct['model'] == model
            pick_default = df_correct['search_type'] == SearchType.DEFAULT.name
            pick_random = df_correct['search_type'] == SearchType.RANDOM.name

            if len(df_correct[correct_model & pick_default]) == 1:
                default_value_val = df_correct[correct_model & pick_default]['score_val_mean'].item()
                default_value_test = df_correct[correct_model & pick_default]['score_test_mean'].item()
            else:
                sweep.logger.warning(f"No default value found for model {model} on dataset {dataset_name}. We will assume 0.")
                default_value_val = 0
                default_value_test = 0
            
            random_values_val = df_correct[correct_model & pick_random]['score_val_mean'].values
            random_values_test = df_correct[correct_model & pick_random]['score_test_mean'].values

            sequences = create_random_sequences(
                default_value_val = default_value_val, 
                default_value_test = default_value_test,
                random_values_val = random_values_val,
                random_values_test = random_values_test,
                sequence_length = sweep.plotting.n_runs,
                n_shuffles = sweep.plotting.n_random_shuffles
            )
            sequences = sequences.clip(min=0)

            sequences_all.append(sequences)

            sequence_mean = np.mean(sequences, axis=0)
            sequence_lower_bound = np.quantile(sequences, q=1-sweep.plotting.confidence_bound, axis=0)
            sequence_upper_bound = np.quantile(sequences, q=sweep.plotting.confidence_bound, axis=0)

            ax.plot(sequence_mean, label=model, linewidth=6)
            ax.fill_between(
                x=np.arange(len(sequence_mean)), 
                y1=sequence_lower_bound, 
                y2=sequence_upper_bound, 
                alpha=0.2
            )

        sequences_stack = np.stack(sequences_all, axis=0)   # [models, n_shuffles, sequence_length]

        ax.set_title(dataset_name)
        ax.title.set_size(30)
        ax.set_xlabel("Number of runs")
        ax.xaxis.label.set_size(30)
        ax.set_ylabel("Test score")
        ax.yaxis.label.set_size(30)
        ax.tick_params(axis='both', which='major', labelsize=15)

        min_y = np.quantile(sequences_stack, q=0.005)
        max_y = np.quantile(sequences_stack, q=0.995)
        spread = max_y - min_y
        min_y = min_y - 0.1 * spread
        max_y = max_y + 0.1 * spread

        ax.set_ylim([min_y, max_y])
        ax.set_xscale('log')
        ax.set_xlim([1, sweep.plotting.n_runs])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))



    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=50)
    fig.tight_layout(pad=2.0, rect=[0, 0.07, 1, 0.92])
    fig.suptitle(f"Test Score for all datasets of size {sweep.dataset_size.name} \n with {sweep.feature_type.name} features on the {sweep.task.name} task", fontsize=40)
    fig.savefig(sweep.sweep_dir / "dataset_plots.png")