import numpy as np
import pandas as pd
from tabularbench.core.enums import ModelName, SearchType
from tabularbench.results.scores_min_max import scores_min_max

from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep



def create_random_sequences_from_df(cfg: ConfigBenchmarkSweep, df: pd.DataFrame) -> np.ndarray: 
    """
    For a given dataframe with results, we create random hpo sequences.

    :returns:
        sequences_all: np.ndarray of shape (n_models, n_datasets, n_shuffles, n_runs)
    """
    
    models = cfg.config_plotting.benchmark_model_names + [ModelName.PLACEHOLDER]

    n_models = len(models)
    n_datasets = len(cfg.openml_dataset_ids_to_use)
    n_shuffles = cfg.config_plotting.n_random_shuffles
    n_runs = cfg.config_plotting.n_runs
    
    sequences_all = np.zeros((n_models, n_datasets, n_shuffles, n_runs))

    for dataset_i, openml_dataset_id in enumerate(cfg.openml_dataset_ids_to_use):

        df_dataset = df[ df['openml_dataset_id'] == openml_dataset_id ]

        for model_i, model in enumerate(models):

            df_model = df_dataset[ df_dataset['model'] == model.name ]

            if model == ModelName.PLACEHOLDER and cfg.search_type == SearchType.DEFAULT:
                # make a confidence interval around the default instead of doing hpo
                sequences_all[model_i, dataset_i, :, :] = compute_default_sequences_for_model(cfg, df_model)
            else:
                sequences_all[model_i, dataset_i, :, :] = compute_random_sequences_for_model(cfg, df_model, model, openml_dataset_id)

    return sequences_all


def compute_default_sequences_for_model(cfg: ConfigBenchmarkSweep, df_model: pd.DataFrame) -> np.ndarray:
    """
    Fake sequence that is just the default value.
    """

    n_shuffles = cfg.config_plotting.n_random_shuffles

    df_model_default = df_model[ df_model['search_type'] == SearchType.DEFAULT.name ]
    results = df_model_default['score_test_mean'].values

    random_index = np.random.randint(0, len(results), size=(n_shuffles,))
    sequences = results[random_index, None]
    sequences = sequences.clip(min=0)

    return sequences


def compute_random_sequences_for_model(cfg: ConfigBenchmarkSweep, df_model: pd.DataFrame, model: str, openml_dataset_id: int) -> np.ndarray:

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

    return sequences


def normalize_sequences(cfg: ConfigBenchmarkSweep, sequences_all: np.ndarray) -> np.ndarray:
    """
    Normalizes the sequences to be normalized test scores between 0 and 1.
    """

    sequences_normalized = np.zeros_like(sequences_all)
    for dataset_i in range(sequences_all.shape[1]):
        score_min, score_max = scores_min_max(cfg, cfg.openml_dataset_ids_to_use[dataset_i])
        normalized = (sequences_all[:, dataset_i, :, :] - score_min).clip(min=0) / (score_max - score_min)
        sequences_normalized[:, dataset_i, :, :] = normalized
    return sequences_normalized



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

