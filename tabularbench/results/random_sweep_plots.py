

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabularbench.core.enums import SearchType, Task

from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.paths_and_filenames import RESULTS_FILE_NAME
from tabularbench.sweeps.sweep_config import SweepConfig


def make_random_sweep_plots(sweep: SweepConfig):

    make_separate_dataset_plots(sweep)
    make_combined_dataset_plot(sweep)



def make_combined_dataset_plot(sweep: SweepConfig):

    df_bench = get_benchmark_csv_reformatted()

    df_cur = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)
    df_cur['model'] = sweep.model_plot_name

    df = pd.concat([df_bench, df_cur], ignore_index=True)


    models = df['model'].unique().tolist()
    if 'HistGradientBoostingTree' in models:
        models.remove('HistGradientBoostingTree')

    sequences_all = np.zeros((len(models), len(sweep.openml_dataset_names), sweep.plotting.n_random_shuffles, sweep.plotting.n_runs))

    for dataset_i, dataset_name in enumerate(sorted(sweep.openml_dataset_names)):

        score_min, score_max = scores_min_max(sweep, df_bench, dataset_name)

        correct_dataset = df['openml_dataset_name'] == dataset_name
        correct_dataset_size = df['dataset_size'] == sweep.dataset_size.name
        correct_feature_type = df['feature_type'] == sweep.feature_type.name
        correct_task = df['task'] == sweep.task.name

        correct_all = correct_dataset & correct_dataset_size & correct_feature_type & correct_task
        df_correct = df.loc[correct_all]

        for model_i, model in enumerate(models):

            correct_model = df_correct['model'] == model
            pick_default = df_correct['search_type'] == SearchType.DEFAULT.name
            pick_random = df_correct['search_type'] == SearchType.RANDOM.name

            default_value_val = df_correct[correct_model & pick_default]['score_val_mean'].item()
            default_value_test = df_correct[correct_model & pick_default]['score_test_mean'].item()
            
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

            sequences_normalized = (sequences - score_min).clip(min=0) / (score_max - score_min)

            sequences_all[model_i, dataset_i, :, :] = sequences_normalized


    sequences_all = np.mean(sequences_all, axis=1)   # [models, sequence_length, n_shuffles]
    
    fig, ax = plt.subplots(figsize=(25, 25))

    for model_i, model in enumerate(models):

        sequence_mean = np.mean(sequences_all[model_i, :, :], axis=0)
        sequence_lower_bound = np.quantile(sequences_all[model_i, :, :], q=1-sweep.plotting.confidence_bound, axis=0)
        sequence_upper_bound = np.quantile(sequences_all[model_i, :, :], q=sweep.plotting.confidence_bound, axis=0)

        ax.plot(sequence_mean, label=model, linewidth=12)
        ax.fill_between(
            x=np.arange(len(sequence_mean)), 
            y1=sequence_lower_bound, 
            y2=sequence_upper_bound, 
            alpha=0.2
        )

    ax.set_title(f"Averaged Normalized Test Score for all datasets of size {sweep.dataset_size.name} \n with {sweep.feature_type.name} features on the {sweep.task.name} task", fontsize=40)
    ax.title.set_size(40)
    ax.set_xlabel("Number of runs")
    ax.xaxis.label.set_size(40)
    ax.set_ylabel("Normalized Test score")
    ax.yaxis.label.set_size(40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    ax.set_xscale('log')
    ax.set_xlim([1, sweep.plotting.n_runs])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=30)
    fig.savefig(sweep.sweep_dir / "combined_dataset_plot.png")



def make_separate_dataset_plots(sweep: SweepConfig):

    df_bench = get_benchmark_csv_reformatted()
    
    df_cur = pd.read_csv(sweep.sweep_dir / RESULTS_FILE_NAME)
    df_cur['model'] = sweep.model_plot_name

    df = pd.concat([df_bench, df_cur], ignore_index=True)

    num_horizontal_subplots = math.ceil(math.sqrt(len(sweep.openml_dataset_names)))
    fig, axs = plt.subplots(nrows=num_horizontal_subplots, ncols=num_horizontal_subplots, figsize=(25, 25))
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

            default_value_val = df_correct[correct_model & pick_default]['score_val_mean'].item()
            default_value_test = df_correct[correct_model & pick_default]['score_test_mean'].item()
            
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
        ax.title.set_size(20)
        ax.set_xlabel("Number of runs")
        ax.xaxis.label.set_size(20)
        ax.set_ylabel("Test score")
        ax.yaxis.label.set_size(20)
        ax.tick_params(axis='both', which='major', labelsize=15)

        min_y = np.quantile(sequences_stack, q=0.005)
        max_y = np.quantile(sequences_stack, q=0.995)
        ax.set_ylim([min_y, max_y])
        ax.set_xscale('log')
        ax.set_xlim([1, sweep.plotting.n_runs])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))



    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=30)
    fig.suptitle(f"Test Score for all datasets of size {sweep.dataset_size.name} \n with {sweep.feature_type.name} features on the {sweep.task.name} task", fontsize=40)
    fig.savefig(sweep.sweep_dir / "dataset_plots.png")



def scores_min_max(sweep: SweepConfig, df: pd.DataFrame, dataset_name: str) -> tuple[float, float]:
    """
    Based on the benchmark results, we normalize the scores of the sweep.
    Returns the min and max scores to normalize with
    """

    models = df['model'].unique().tolist()
    models = [model for model in models if model in sweep.plotting.benchmark_models]

    correct_model = df['model'].isin(models)
    correct_dataset = df['openml_dataset_name'] == dataset_name
    correct_dataset_size = df['dataset_size'] == sweep.dataset_size.name
    correct_feature_type = df['feature_type'] == sweep.feature_type.name
    correct_task = df['task'] == sweep.task.name

    correct_all = correct_dataset & correct_dataset_size & correct_feature_type & correct_task & correct_model
    df_correct = df.loc[correct_all]

    match sweep.task:
        case Task.REGRESSION:
            score_min = df_correct['score_test_mean'].quantile(0.50)
        case Task.CLASSIFICATION:
            score_min = df_correct['score_test_mean'].quantile(0.10)

    score_max = df_correct['score_test_mean'].max()

    sweep.logger.info(f"For dataset {dataset_name}, we will normalize with min {score_min:.4f} and max {score_max:.4f}")

    return score_min, score_max








def create_random_sequences(
    default_value_val: float, 
    default_value_test: float,
    random_values_val: np.ndarray, 
    random_values_test: np.ndarray,
    sequence_length: int,
    n_shuffles: int
):
    """
    Makes random test sequences.
    Let random_values_val and random_values_test be arrays of shape (n_runs,), which are the scores of the sweep.
    We are interested what happens if we would have executed this sweep in a different order.
    We pick sequence_length random values from random_values_val and random_values_test, randomize the order (with replacement), and prepend the default values.
    We track the running-best validation score, and return the matching test score for each sequence.
    The number of sequences is n_shuffles.

    returns:
        best_test_score: np.ndarray of shape (n_shuffles, sequence_length)
    """

    
    assert len(random_values_val) == len(random_values_test), "The number of random values for val and test must be the same"

    if len(random_values_val) == 0:
        # We consider default runs (no random values) as a drawn horizontal line
        return np.tile(default_value_test, (n_shuffles, sequence_length))

    random_values = np.concatenate([random_values_val[None, :], random_values_test[None, :]], axis=0)
    default_values = np.array([default_value_val, default_value_test])

    random_index = np.random.randint(0, len(random_values_val), size=(n_shuffles, sequence_length-1))

    random_sequences = random_values[:, random_index]
    sequences = np.concatenate([np.tile(default_values[:, None], (1, n_shuffles))[:, :, None], random_sequences], axis=2)

    best_validation_score = np.maximum.accumulate(sequences[0, :, :], axis=1)
    diff = best_validation_score[:, :-1] < best_validation_score[:, 1:]
    diff_prepend_zeros = np.concatenate([np.zeros((n_shuffles, 1), dtype=bool), diff], axis=1)
    best_validation_idcs = diff_prepend_zeros * np.arange(sequence_length)[None, :]
    best_validation_idcs = np.maximum.accumulate(best_validation_idcs, axis=1)

    best_test_score = sequences[1, np.arange(n_shuffles)[:, None], best_validation_idcs ]
    
    return best_test_score
    


if __name__ == '__main__':

    seq = create_random_sequences(5, 6, np.array([3, 8, 3, 6, 3, 3, 7, 4, 3]), np.array([4, 5, 3, 7, 3, 4, 7, 3, 4]), 5, 3)
    print(seq)
    pass

