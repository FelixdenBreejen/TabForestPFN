from pathlib import Path
import numpy as np
import openml
import xarray as xr
from sklearn.preprocessing import LabelEncoder
from tabularbench.core.enums import DatasetSize

from tabularbench.utils.paths_and_filenames import PATH_TO_DATA_SPLIT, PATH_TO_OPENML_DATASETS


class OpenmlDatafile():

    def __init__(self, openml_dataset_id: int, dataset_size: DatasetSize):

        self.openml_dataset_id = openml_dataset_id
        self.dataset_size = dataset_size

        self.data_path = Path(f"{PATH_TO_OPENML_DATASETS}/{openml_dataset_id}_{dataset_size.name}.nc")
        self.data_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.data_path.exists():
            self.download_dataset()

        self.get_dataset_from_disk()


    def get_dataset_from_disk(self):
            
        self.ds = xr.open_dataset(self.data_path)
        self.x = self.ds['x'].values
        self.y = self.ds['y'].values
        self.categorical_indicator = self.ds['categorical_indicator'].values
        self.attribute_names = self.ds['attribute_names'].values


    def download_dataset(self):

        dataset = openml.datasets.get_dataset(self.openml_dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)

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

        split_train, split_val, split_test = self.get_splits(n_observations)
        n_splits = split_train.shape[1]
        

        self.ds = xr.Dataset(
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
                'openml_dataset_id': self.openml_dataset_id,
                'openml_dataset_name': dataset.name,
            }
        )
    
        self.ds.to_netcdf(self.data_path)
        

    def get_splits(self, n_observations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        splits = np.load(PATH_TO_DATA_SPLIT, allow_pickle=True).item()
        split = splits[self.openml_dataset_id][self.dataset_size.value]
        n_splits = len(split)

        split_train = np.zeros((n_observations, n_splits), dtype=bool)
        split_val = np.zeros((n_observations, n_splits), dtype=bool)
        split_test = np.zeros((n_observations, n_splits), dtype=bool)

        for i in range(n_splits):
            split_train[split[i]['train'], i] = True
            split_val[split[i]['val'], i] = True
            split_test[split[i]['test'], i] = True

        return split_train, split_val, split_test



