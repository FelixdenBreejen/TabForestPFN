from pathlib import Path
import numpy as np
import xarray as xr
from loguru import logger

from sklearn.preprocessing import LabelEncoder
import openml
from tabularbench.core.enums import DatasetSize

from tabularbench.utils.paths_and_filenames import PATH_TO_DATA_SPLIT, PATH_TO_OPENML_DATASETS


def main():

    openml_ids = get_openml_ids()
    for openml_id in openml_ids:
        logger.info(f"Downloading dataset {openml_id}...")
        download_dataset(openml_id)
        logger.info(f"Downloaded dataset {openml_id}.")


def get_openml_ids() -> list[int]:

    path = Path('tabularbench/data/whytrees_benchmark_openml_ids.txt')

    with open(path, 'r') as f:
        ids = f.readlines()

    return [int(id) for id in ids]



def download_dataset(openml_id: int) -> None:


    dataset = openml.datasets.get_dataset(openml_id, download_data=True, download_qualities=False, download_features_meta_data=False)

    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute
    )
    x = X.to_numpy().astype(np.float32)
    y = y.to_numpy()

    if y.dtype == np.dtype('O'):
        y = LabelEncoder().fit_transform(y)

    assert categorical_indicator is not None
    categorical_indicator = np.array(categorical_indicator).astype(bool)
    attribute_names = np.array(attribute_names)

    n_observations = x.shape[0]
    n_features = x.shape[1]

    for dataset_size in [DatasetSize.MEDIUM, DatasetSize.LARGE]:
        
        split_train, split_val, split_test = get_splits(openml_id, dataset_size, n_observations)
        n_splits = split_train.shape[1]
        

        ds = xr.Dataset(
            data_vars={
                'x': (['observation', 'feature'], x),
                'y': (['observation'], y),
                'split_index_train': (['observation', 'split'], split_train),
                'split_index_val': (['observation', 'split'], split_val),
                'split_index_test': (['observation', 'split'], split_test),
                'categorical_indicator': (['feature'], categorical_indicator),
                'attribute_names': (['feature'], attribute_names)
            },
            coords={
                'observation': np.arange(n_observations),
                'feature': np.arange(n_features),
                'split': np.arange(n_splits),
            },
            attrs={
                'openml_dataset_id': openml_id,
                'openml_dataset_name': dataset.name,
            }
        )

        p = Path(PATH_TO_OPENML_DATASETS) / f'whytrees_{openml_id}_{dataset_size.name}.nc'
        ds.to_netcdf(p)
    

def get_splits(openml_id: int, dataset_size: int, n_observations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    splits = np.load(PATH_TO_DATA_SPLIT, allow_pickle=True).item()
    split = splits[openml_id][dataset_size]
    n_splits = len(split)

    split_train = np.zeros((n_observations, n_splits), dtype=bool)
    split_val = np.zeros((n_observations, n_splits), dtype=bool)
    split_test = np.zeros((n_observations, n_splits), dtype=bool)

    for i in range(n_splits):
        split_train[split[i]['train'], i] = True
        split_val[split[i]['val'], i] = True
        split_test[split[i]['test'], i] = True

    return split_train, split_val, split_test


if __name__ == "__main__":
    main()