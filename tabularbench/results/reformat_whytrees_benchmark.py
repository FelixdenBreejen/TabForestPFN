import functools
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm import tqdm

from tabularbench.core.enums import BenchmarkName, DatasetSize, DataSplit, FeatureType, ModelName, SearchType, Task
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.utils.paths_and_filenames import (PATH_TO_WHYTREES_BENCH_RESULTS,
                                                    PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)


def reformat_whytrees_benchmark():

    logger.info("Reformatting whytrees benchmark started")

    path = Path(PATH_TO_WHYTREES_BENCH_RESULTS)

    assert path.exists(), f"File {path} does not exist, did you download the benchmark?"
    logger.info(f"Reformatting whytrees benchmark from {path}")
    
    logger.info(f"Reading csv file...")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Dropping rows with missing values...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['max_train_samples', 'data__regression', 'data__categorical', 'mean_train_score', 'mean_val_score', 'mean_test_score'])

    logger.info(f"Retrieving benchmark names...")
    df['benchmark_name'] = df.apply(get_benchmark_name, axis=1)
    logger.info(f"Retrieving openml dataset ids...")
    df['openml_dataset_id'] = df.apply(get_openml_id, axis=1)

    logger.info(f"Building xarray dataset...")
    coords_dict = make_coords_dict(df)
    ds = xr.Dataset(
        data_vars=make_data_vars_dict_with_empty_initialization(df, coords_dict),
        coords=coords_dict,
        attrs=make_attr_dict(df)
    )

    logger.info(f"Populating xarray dataset...")
    populate_xarray_dataset(df, ds)

    ds.to_netcdf(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)
    logger.info(f"Saved xarray dataset to {PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED}")


def make_coords_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:

    model_names = df['model_name'].unique()
    model_names = [get_model_name(model_name).name for model_name in model_names]
    openml_dataset_ids = np.sort(df['openml_dataset_id'].unique())
    n_runs_max = df.groupby(['model_name', 'openml_dataset_id']).size().max()
    n_cv_splits = get_max_n_cv_splits_of_all_datasets(df)

    return {
        'model_name': model_names,
        'openml_dataset_id': openml_dataset_ids,
        'run_id': np.arange(n_runs_max),
        'cv_split': np.arange(n_cv_splits),
        'data_split': [data_split.value for data_split in DataSplit],
    }


def make_data_vars_dict_with_empty_initialization(df: pd.DataFrame, coords_dict: dict[str, np.ndarray]) -> dict[str, tuple]:

    n_models = len(coords_dict['model_name'])
    n_datasets = len(coords_dict['openml_dataset_id'])
    n_runs_max = len(coords_dict['run_id'])
    n_cv_splits = len(coords_dict['cv_split'])
    n_data_splits = len(coords_dict['data_split'])

    score = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    search_type = np.full((n_models, n_datasets, n_runs_max), "", dtype=object)
    task = np.full((n_datasets), "", dtype=object)
    feature_type = np.full((n_datasets), "", dtype=object)
    dataset_size = np.full((n_datasets), "", dtype=object)
    benchmark_name = np.full((n_datasets), "", dtype=object)
    openml_dataset_name = np.full((n_datasets), "", dtype=object)
    runs_actual = np.full((n_models, n_datasets), 0, dtype=int)
    cv_splits_actual = np.full((n_datasets), 0, dtype=int)

    return {
        'score': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], score),
        'search_type': (['model_name', 'openml_dataset_id', 'run_id'], search_type),
        'task': (['openml_dataset_id'], task),
        'feature_type': (['openml_dataset_id'], feature_type),
        'dataset_size': (['openml_dataset_id'], dataset_size),
        'benchmark_name': (['openml_dataset_id'], benchmark_name),
        'openml_dataset_name': (['openml_dataset_id'], openml_dataset_name),
        'runs_actual': (['model_name', 'openml_dataset_id'], runs_actual),
        'cv_splits_actual': (['openml_dataset_id'], cv_splits_actual),
    }


def get_max_n_cv_splits_of_all_datasets(df: pd.DataFrame) -> list[int]:
    return int((df['train_scores'].str.count(',')+1).max())


def make_attr_dict(df: pd.DataFrame) -> dict[str, str]:

    return {
        'description': (
            'Reformatted whytrees benchmark, from the paper '
            '"Why do tree-based models still outperform deep learning on tabular data?" by Grinsztasjn et al. (2022)'
        ),
        'details': (
            'Score is the same as accuracy or r2, depending on the task (classification vs regression).'
            'Losses are not given by the benchmark dataset.'
        )
    }


def populate_xarray_dataset(df: pd.DataFrame, ds: xr.Dataset) -> None:

    for _, row in tqdm(df.iterrows(), total=len(df)):

        model_name = get_model_name(row['model_name']).name
        openml_dataset_id = row['openml_dataset_id']

        search_type = SearchType.RANDOM if row['hp'] == 'random' else SearchType.DEFAULT
        task = Task.REGRESSION if row['data__regression'] else Task.CLASSIFICATION
        feature_type = FeatureType.CATEGORICAL if row['data__categorical'] else FeatureType.NUMERICAL
        dataset_size = DatasetSize(row['max_train_samples'])
        openml_dataset_name = row['data__keyword']
        benchmark_name = BenchmarkName(row['benchmark_name'])

        run_id = ds['runs_actual'].loc[model_name, openml_dataset_id].item()
        ds['runs_actual'].loc[model_name, openml_dataset_id] += 1

        score, n_cv_splits_actual = get_scores_with_n_cv_splits_actual(row)
        cv_split_slice = slice(0, n_cv_splits_actual - 1) # xarray slice is inclusive
        
        ds['score'].loc[model_name, openml_dataset_id, run_id, cv_split_slice] = score
        ds['search_type'].loc[model_name, openml_dataset_id, run_id] = search_type.name
        ds['task'].loc[openml_dataset_id] = task.name
        ds['feature_type'].loc[openml_dataset_id] = feature_type.name
        ds['dataset_size'].loc[openml_dataset_id] = dataset_size.name
        ds['benchmark_name'].loc[openml_dataset_id] = benchmark_name.name
        ds['openml_dataset_name'].loc[openml_dataset_id] = openml_dataset_name

        if run_id == 0:
            ds['cv_splits_actual'].loc[openml_dataset_id] = n_cv_splits_actual
        else:
            assert n_cv_splits_actual == ds['cv_splits_actual'].loc[openml_dataset_id]
    

def get_scores_with_n_cv_splits_actual(row: pd.Series) -> tuple[float, int]:

    if row['val_scores'] is np.nan or row['test_scores'] is np.nan:
        train_scores = [row['mean_train_score']]
        val_scores = [row['mean_val_score']]
        test_scores = [row['mean_test_score']]
    else:
        train_scores = [float(score) for score in row['train_scores'].strip('[]').split(',')]
        val_scores = [float(score) for score in row['val_scores'].strip('[]').split(',')]
        test_scores = [float(score) for score in row['test_scores'].strip('[]').split(',')]

    score = np.array([train_scores, val_scores, test_scores]).T
    n_cv_splits_actual = len(val_scores)

    return score, n_cv_splits_actual


def get_benchmark_name(row: pd.Series) -> str:

    task = Task.REGRESSION if row['data__regression'] else Task.CLASSIFICATION
    feature_type = FeatureType.CATEGORICAL if row['data__categorical'] else FeatureType.NUMERICAL
    dataset_size = DatasetSize(row['max_train_samples'])

    match (task, dataset_size, feature_type):
        case (Task.CLASSIFICATION, DatasetSize.LARGE, FeatureType.CATEGORICAL):
            benchmark_name = BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE
        case (Task.CLASSIFICATION, DatasetSize.LARGE, FeatureType.NUMERICAL):
            benchmark_name = BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE
        case (Task.CLASSIFICATION, DatasetSize.MEDIUM, FeatureType.CATEGORICAL):
            benchmark_name = BenchmarkName.CATEGORICAL_CLASSIFICATION
        case (Task.CLASSIFICATION, DatasetSize.MEDIUM, FeatureType.NUMERICAL):
            benchmark_name = BenchmarkName.NUMERICAL_CLASSIFICATION
        case (Task.REGRESSION, DatasetSize.LARGE, FeatureType.CATEGORICAL):
            benchmark_name = BenchmarkName.CATEGORICAL_REGRESSION_LARGE
        case (Task.REGRESSION, DatasetSize.LARGE, FeatureType.NUMERICAL):
            benchmark_name = BenchmarkName.NUMERICAL_REGRESSION_LARGE
        case (Task.REGRESSION, DatasetSize.MEDIUM, FeatureType.CATEGORICAL):
            benchmark_name = BenchmarkName.CATEGORICAL_REGRESSION
        case (Task.REGRESSION, DatasetSize.MEDIUM, FeatureType.NUMERICAL):
            benchmark_name = BenchmarkName.NUMERICAL_REGRESSION
    
    return benchmark_name.value


def get_openml_id(row: pd.Series) -> int:

    benchmark_name_str = row['benchmark_name']
    benchmark_name = BenchmarkName(benchmark_name_str)
    benchmark = BENCHMARKS[benchmark_name]

    dataset_name = row['data__keyword']

    assert dataset_name in benchmark.openml_dataset_names, f"Dataset {dataset_name} not in benchmark {benchmark_name}"
    index = benchmark.openml_dataset_names.index(dataset_name)
    dataset_id = benchmark.openml_dataset_ids[index]
    return dataset_id


def get_model_name(str: str) -> ModelName:
    
    model_name_dict = {
        'MLP': ModelName.MLP,
        'FT Transformer': ModelName.FT_TRANSFORMER,
        'Resnet': ModelName.RESNET,
        'SAINT': ModelName.SAINT,
        'RandomForest': ModelName.RANDOM_FOREST,
        'XGBoost': ModelName.XGBOOST,
        'GradientBoostingTree': ModelName.GRADIENT_BOOSTING_TREE,
        'HistGradientBoostingTree': ModelName.HIST_GRADIENT_BOOSTING_TREE,
    }

    return model_name_dict[str]


@functools.lru_cache(maxsize=1)
def get_whytrees_benchmark_reformatted() -> xr.Dataset:

    if not Path(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED).exists():
        raise FileNotFoundError(f"File {PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED} does not exist, did you run reformat_whytrees_benchmark()?")

    return xr.open_dataset(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)

if __name__ == "__main__":
    reformat_whytrees_benchmark()