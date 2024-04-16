from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.utils.paths_and_filenames import (DATASETS_TABZILLA_GLOB, DATASETS_WHYTREES_GLOB,
                                                    PATH_TO_OPENML_DATASETS)


def create_metadata(benchmark_origin: BenchmarkOrigin):

    match benchmark_origin:
        case BenchmarkOrigin.TABZILLA:
            list_of_paths = list(Path(PATH_TO_OPENML_DATASETS).glob(DATASETS_TABZILLA_GLOB))
        case BenchmarkOrigin.WHYTREES:
            list_of_paths = list(Path(PATH_TO_OPENML_DATASETS).glob(DATASETS_WHYTREES_GLOB))
    
    return create_metadata_(list_of_paths)


def create_metadata_(list_of_dataset_paths: list[Path]) -> pd.DataFrame:

    dss = [xr.open_dataset(path) for path in list_of_dataset_paths]

    metadata = []

    for ds in dss:

        metadata.append({
            'openml_dataset_id': ds.attrs['openml_dataset_id'],
            'openml_dataset_name': ds.attrs['openml_dataset_name'],
            'n_observations': ds.sizes['observation'],
            'n_train': ds['split_index_train'].sel(split=0).sum().item(),
            'n_val': ds['split_index_val'].sel(split=0).sum().item(),
            'n_test': ds['split_index_test'].sel(split=0).sum().item(),
            'n_features': ds.sizes['feature'],
            'n_splits': ds.sizes['split'],
            'n_classes': len(np.unique(ds['y']))

        })

    df = pd.DataFrame(metadata)
    df.set_index('openml_dataset_id', inplace=True)
    df.sort_index(inplace=True)

    return df