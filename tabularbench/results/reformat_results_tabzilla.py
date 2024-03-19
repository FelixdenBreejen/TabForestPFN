from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from tqdm import tqdm

from tabularbench.core.enums import DataSplit, ModelName, SearchType
from tabularbench.core.model_class import get_model_class
from tabularbench.results.reformat_results_whytrees import get_model_name
from tabularbench.utils.paths_and_filenames import (PATH_TO_TABZILLA_BENCH_RESULTS,
                                                    PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED)

TABZILLA_CV_SPLIT_COUNT = 10

def reformat_results_tabzilla():

    logger.info("Reformatting tabzilla benchmark started")

    path = Path(PATH_TO_TABZILLA_BENCH_RESULTS)

    assert path.exists(), f"File {path} does not exist, did you download the benchmark?"
    logger.info(f"Reformatting tabzilla benchmark from {path}")
    
    logger.info(f"Reading csv file...")
    df = pd.read_csv(path, low_memory=False)

    logger.info(f"Handling missing values...")
    df.loc[df['AUC__val'].isna(), 'AUC__val'] = 0
    df.loc[df['AUC__test'].isna(), 'AUC__test'] = 0
    assert df.isna().sum().sum() == 0, "There are still missing values in the dataframe"

    logger.info(f"Creating openml dataset ids and names...")
    df['openml_dataset_name'] = df['dataset_name'].str.split('__', n=2).str[1]
    df['openml_dataset_id'] = df['dataset_name'].str.split('__', n=2).str[2].astype(int)

    logger.info(f"Building xarray dataset...")
    coords_dict = make_coords_dict(df)
    ds = xr.Dataset(
        data_vars=make_data_vars_dict_with_empty_initialization(df, coords_dict),
        coords=coords_dict,
        attrs=make_attr_dict(df)
    )

    logger.info(f"Populating xarray dataset...")
    populate_xarray_dataset(df, ds)

    ds.to_netcdf(PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED)
    logger.info(f"Saved xarray dataset to {PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED}")


def make_coords_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:

    model_names = df['alg_name'].unique()
    model_names = [get_model_name(model_name).name for model_name in model_names]
    openml_dataset_ids = np.sort(df['openml_dataset_id'].unique())
    n_runs_max = df.groupby(['alg_name', 'dataset_fold_id']).size().max()
    n_cv_splits = TABZILLA_CV_SPLIT_COUNT   # This paper uses 10-fold cross-validation

    return {
        'model_name': model_names,
        'openml_dataset_id': openml_dataset_ids,
        'run_id': np.arange(n_runs_max),
        'cv_split': np.arange(n_cv_splits),
        'data_split': [data_split.name for data_split in DataSplit],
    }


def make_data_vars_dict_with_empty_initialization(df: pd.DataFrame, coords_dict: dict[str, np.ndarray]) -> dict[str, tuple]:

    n_models = len(coords_dict['model_name'])
    n_datasets = len(coords_dict['openml_dataset_id'])
    n_runs_max = len(coords_dict['run_id'])
    n_cv_splits = len(coords_dict['cv_split'])
    n_data_splits = len(coords_dict['data_split'])

    score = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    search_type = np.full((n_models, n_datasets, n_runs_max), "", dtype=object)
    log_loss = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    auc = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    acc = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    f1 = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)
    time_training = np.full((n_models, n_datasets, n_runs_max, n_cv_splits), np.nan)
    time_eval = np.full((n_models, n_datasets, n_runs_max, n_cv_splits, n_data_splits), np.nan)

    openml_dataset_name = np.full((n_datasets), "", dtype=object)
    runs_actual = np.full((n_models, n_datasets), 0, dtype=int)
    cv_splits_actual = np.full((n_datasets), TABZILLA_CV_SPLIT_COUNT, dtype=int)
    model_class = np.full((n_models), "", dtype=object)

    return {
        'score': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], score),
        'search_type': (['model_name', 'openml_dataset_id', 'run_id'], search_type),
        'log_loss': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], log_loss),
        'auc': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], auc),
        'acc': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], acc),
        'f1': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], f1),
        'time_training': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split'], time_training),
        'time_eval': (['model_name', 'openml_dataset_id', 'run_id', 'cv_split', 'data_split'], time_eval),
        'openml_dataset_name': (['openml_dataset_id'], openml_dataset_name),
        'runs_actual': (['model_name', 'openml_dataset_id'], runs_actual),
        'cv_splits_actual': (['openml_dataset_id'], cv_splits_actual),
        'model_class': (['model_name'], model_class),
    }


def make_attr_dict(df: pd.DataFrame) -> dict[str, str]:

    return {
        'description': (
            'Reformatted tabzilla benchmark, from the paper '
            '"When Do Neural Nets Outperform Boosted Trees on Tabular Data?" by McElfresh et al. (2023)'
        ),
        'details': (
            'Score is the same as accuracy.'
            'Task is Classification for the whole dataset, even though the paper itself makes a distinction between binary and classification.'
        )
    }


def populate_xarray_dataset(df: pd.DataFrame, ds: xr.Dataset) -> None:

    runs_actual = xr.DataArray(
        data = np.zeros((len(ds['model_name']), len(ds['openml_dataset_id']), len(ds['cv_split'])), dtype=int),
        coords = { 
            'model_name': ds['model_name'].values, 
            'openml_dataset_id': ds['openml_dataset_id'].values, 
            'cv_split': ds['cv_split'].values 
        }
    )

    for _, row in tqdm(df.iterrows(), total=len(df)):

        model_name = get_model_name(row['alg_name'])
        openml_dataset_id = row['openml_dataset_id']

        search_type = SearchType.DEFAULT if row['hparam_source'] == 'default' else SearchType.RANDOM
        openml_dataset_name = row['openml_dataset_name']

        cv_split = int(row['dataset_fold_id'].split('fold_')[1])

        run_id = runs_actual.sel(model_name=model_name.name, openml_dataset_id=openml_dataset_id, cv_split=cv_split).item()
        runs_actual.loc[model_name.name, openml_dataset_id, cv_split] += 1
        ds['runs_actual'].loc[model_name.name, openml_dataset_id] = runs_actual.sel(model_name=model_name.name, openml_dataset_id=openml_dataset_id).max()
        
        ds['score'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['Accuracy__train'], row['Accuracy__val'], row['Accuracy__test'] ]
        ds['search_type'].loc[model_name.name, openml_dataset_id, run_id] = search_type.name
        ds['log_loss'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['Log Loss__train'], row['Log Loss__val'], row['Log Loss__test'] ]
        ds['auc'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['AUC__train'], row['AUC__val'], row['AUC__test'] ]
        ds['acc'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['Accuracy__train'], row['Accuracy__val'], row['Accuracy__test'] ]
        ds['f1'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['F1__train'], row['F1__val'], row['F1__test'] ]
        ds['time_training'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = row['training_time']
        ds['time_eval'].loc[model_name.name, openml_dataset_id, run_id, cv_split] = [ row['eval-time__train'], row['eval-time__val'], row['eval-time__test'] ]

        ds['openml_dataset_name'].loc[openml_dataset_id] = openml_dataset_name
        ds['model_class'].loc[model_name.name] = get_model_class(model_name).name


def get_model_name(str: str) -> ModelName:

    model_name_dict = {
        'CatBoost': ModelName.CATBOOST,
        'DecisionTree': ModelName.DECISION_TREE,
        'DeepFM': ModelName.DEEPFM,
        'KNN': ModelName.KNN,
        'LightGBM': ModelName.LIGHTGBM,
        'LinearModel': ModelName.LINEAR_REGRESSION,
        'MLP': ModelName.MLP,
        'RandomForest': ModelName.RANDOM_FOREST,
        'STG': ModelName.STG,
        'SVM': ModelName.SVM,
        'TabNet': ModelName.TABNET,
        'TabTransformer': ModelName.TABTRANSFORMER,
        'VIME': ModelName.VIME,
        'XGBoost': ModelName.XGBOOST,
        'rtdl_MLP': ModelName.MLP_RTDL,
        'rtdl_ResNet': ModelName.RESNET,
        'DANet': ModelName.DANET,
        'NAM': ModelName.NAM,
        'NODE': ModelName.NODE,
        'SAINT': ModelName.SAINT,
        'rtdl_FTTransformer': ModelName.FT_TRANSFORMER,
        'TabPFNModel': ModelName.TABPFN,
    }

    return model_name_dict[str]



if __name__ == "__main__":
    reformat_results_tabzilla()