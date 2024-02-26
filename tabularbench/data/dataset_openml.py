from pathlib import Path
from typing import Iterator
import numpy as np
import xarray as xr
from sklearn.preprocessing import QuantileTransformer

from tabularbench.core.enums import FeatureType, Task



class OpenMLDataset():

    def __init__(self, datafile_path: Path, task: Task):

        self.datafile_path = datafile_path
        self.task = task

        ds = xr.open_dataset(self.datafile_path)
        X = ds['x'].values
        y = ds['y'].values
        categorical_indicator = ds['categorical_indicator'].values

        self.splits_train = ds['split_index_train'].values
        self.splits_val = ds['split_index_val'].values
        self.splits_test = ds['split_index_test'].values
        self.n_splits = ds.dims['split']

        self.X, self.y, self.categorical_indicator = self.do_basic_preprocessing(X, y, categorical_indicator)

        self.n_classes = len(np.unique(self.y))





    def do_basic_preprocessing(self, X, y, categorical_indicator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        has_categorical_features = categorical_indicator.any()
        has_numerical_features = (~categorical_indicator).any()

        if has_categorical_features and has_numerical_features:
            self.feature_type = FeatureType.MIXED
        elif has_categorical_features:
            self.feature_type = FeatureType.CATEGORICAL
        elif has_numerical_features:
            self.feature_type = FeatureType.NUMERICAL
        else:
            raise ValueError("There are neither categorical nor numerical features in the dataset")

        match self.task:
            case Task.CLASSIFICATION:
                y = y.astype(np.int64)
            case Task.REGRESSION:
                y = y.astype(np.float32)

        assert X.shape[0] == y.shape[0], "X and y have different number of samples"
        assert X.shape[1] == categorical_indicator.shape[0], "X and categorical_indicator have different number of features"
        assert len(y.shape) == 1, "y has more than one dimension" 

        return X, y, categorical_indicator
    

    def split_iterator(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        

        for split_i in range(self.n_splits):
            
            train_idcs = self.splits_train[:, split_i]
            val_idcs = self.splits_val[:, split_i]
            test_idcs = self.splits_test[:, split_i]

            x_train = self.X[train_idcs]
            x_val = self.X[val_idcs]
            x_test = self.X[test_idcs]

            y_train = self.y[train_idcs]
            y_val = self.y[val_idcs]
            y_test = self.y[test_idcs]

            categorical_indicator = self.categorical_indicator

            if self.feature_type == FeatureType.NUMERICAL or self.feature_type == FeatureType.MIXED:
                qt = QuantileTransformer(output_distribution="normal")
                x_train[:, ~categorical_indicator] = qt.fit_transform(x_train[:, ~categorical_indicator])
                x_val[:, ~categorical_indicator] = qt.transform(x_val[:, ~categorical_indicator])
                x_test[:, ~categorical_indicator] = qt.transform(x_test[:, ~categorical_indicator])

            yield x_train, x_val, x_test, y_train, y_val, y_test, categorical_indicator



