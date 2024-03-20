from pathlib import Path

import pandas as pd
import xarray as xr


def create_metadata(list_of_dataset_paths: list[Path]) -> pd.DataFrame:

    dss = [xr.open_dataset(path) for path in list_of_dataset_paths]

    metadata = []

    for ds in dss:

        metadata.append({
            'openml_dataset_id': ds.attrs['openml_dataset_id'],
            'openml_dataset_name': ds.attrs['openml_dataset_name'],
            'n_observations': ds.sizes['observation'],
            'n_features': ds.sizes['feature'],
            'n_splits': ds.sizes['split'],
            'n_train': ds['split_index_train'].sel(split=0).sum().item(),
            'n_val': ds['split_index_val'].sel(split=0).sum().item(),
            'n_test': ds['split_index_test'].sel(split=0).sum().item(),
        })

    metadata = pd.DataFrame(metadata)
    metadata.set_index('openml_dataset_id', inplace=True)
    metadata.sort_index(inplace=True)

    return metadata