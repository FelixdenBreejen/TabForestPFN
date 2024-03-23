import numpy as np
import xarray as xr
from loguru import logger

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.core.enums import DataSplit, ModelName, SearchType
from tabularbench.results.dataset_manipulations import average_out_the_cv_split
from tabularbench.results.scores_min_max import scores_min_max


def create_random_sequences_from_dataset(cfg: ConfigBenchmarkSweep, ds: xr.Dataset) -> np.ndarray:
    """
    For a given dataframe with results, we create random hpo sequences.

    :returns:
        sequences_all: np.ndarray of shape (n_models, n_datasets, n_shuffles, n_runs)
    """
    
    models = cfg.plotting.whytrees.benchmark_model_names + [ModelName.PLACEHOLDER]

    n_models = len(models)
    n_datasets = len(cfg.openml_dataset_ids_to_use)
    n_shuffles = cfg.plotting.whytrees.n_random_shuffles
    n_runs = cfg.plotting.whytrees.n_runs
    
    sequences_all = np.zeros((n_models, n_datasets, n_shuffles, n_runs))

    ds = average_out_the_cv_split(ds)

    for dataset_i, openml_dataset_id in enumerate(cfg.openml_dataset_ids_to_use):

        ds_dataset = ds.sel(openml_dataset_id=openml_dataset_id)

        for model_i, model in enumerate(models):

            ds_model = ds_dataset.sel(model_name=model.name)

            if model == ModelName.PLACEHOLDER and cfg.search_type == SearchType.DEFAULT:
                # make a confidence interval around the default instead of doing hpo
                sequences_all[model_i, dataset_i, :, :] = compute_default_sequences_for_model(cfg, ds_model)
            else:
                sequences_all[model_i, dataset_i, :, :] = compute_random_sequences_for_model(cfg, ds_model, model, openml_dataset_id)

    return sequences_all


def compute_default_sequences_for_model(cfg: ConfigBenchmarkSweep, ds_model: xr.Dataset) -> np.ndarray:
    """
    Fake sequence that is just the default value.
    """

    n_shuffles = cfg.plotting.whytrees.n_random_shuffles

    ds = ds_model.sel(data_split=DataSplit.TEST.name)
    results = ds['score'].where(ds_model['search_type'] == SearchType.DEFAULT.name, drop=True).values

    random_index = np.random.randint(0, len(results), size=(n_shuffles,))
    sequences = results[random_index, None]
    sequences = sequences.clip(min=0)

    return sequences


def compute_random_sequences_for_model(cfg: ConfigBenchmarkSweep, ds_model: xr.Dataset, model: str, openml_dataset_id: int) -> np.ndarray:

    ds_default = ds_model['score'].where(ds_model['search_type'] == SearchType.DEFAULT.name, drop=True)
    ds_default_seed_0 = ds_default.where(ds_model['seed'] == cfg.seed, drop=True)

    if ds_default.sizes['run_id'] == 1:
        # If there is one default value, we use that
        default_value_val = ds_default.sel(data_split=DataSplit.VALID.name).item()
        default_value_test = ds_default.sel(data_split=DataSplit.TEST.name).item()
    elif ds_default_seed_0.sizes['run_id'] == 1:
        # If there are multiple default values, we use the one with seed 0
        default_value_val = ds_default_seed_0.sel(data_split=DataSplit.VALID.name).item()
        default_value_test = ds_default_seed_0.sel(data_split=DataSplit.TEST.name).item()
    elif ds_default.sizes['run_id'] == 0:
        logger.warning(f"No default value found for model {model} on dataset {openml_dataset_id}. We will assume 0.")
        default_value_val = 0
        default_value_test = 0
    else:
        raise ValueError(f"More than one default value found for model {model} on dataset {openml_dataset_id}")
    
    ds_random = ds_model['score'].where(ds_model['search_type'] == SearchType.RANDOM.name, drop=True)

    random_values_val = ds_random.sel(data_split=DataSplit.VALID.name).values
    random_values_test = ds_random.sel(data_split=DataSplit.TEST.name).values

    sequences = create_random_sequences(
        default_value_val = default_value_val, 
        default_value_test = default_value_test,
        random_values_val = random_values_val,
        random_values_test = random_values_test,
        sequence_length = cfg.plotting.whytrees.n_runs,
        n_shuffles = cfg.plotting.whytrees.n_random_shuffles
    )
    sequences = sequences.clip(min=0)

    return sequences


def normalize_sequences(cfg: ConfigBenchmarkSweep, sequences_all: np.ndarray) -> np.ndarray:
    """
    Normalizes the sequences to be normalized test scores between 0 and 1.
    """

    sequences_normalized = np.zeros_like(sequences_all)
    
    for dataset_i in range(sequences_all.shape[1]):
        score_min, score_max = scores_min_max(cfg, cfg.openml_dataset_ids_to_use[dataset_i], DataSplit.TEST)
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
        # If there are no random values, we just return the default value
        return np.tile(default_value_test, (n_shuffles, sequence_length))

    random_values = np.concatenate([random_values_val[None, :], random_values_test[None, :]], axis=0)
    default_values = np.array([default_value_val, default_value_test])

    random_index = np.random.randint(0, len(random_values_val), size=(n_shuffles, sequence_length-1))
    random_sequences = random_values[:, random_index]
    
    default_start = np.tile(default_values[:, None], (1, n_shuffles))[:, :, None]
    sequences = np.concatenate([default_start, random_sequences], axis=2)

    # sequences are now and array of shape (2, n_shuffles, n_runs),
    # where the first dimension is val and test, and the second dimension is the shuffles,
    # and the third dimension is the runs. Every run starts with the default value, and then has random HPO values.

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

