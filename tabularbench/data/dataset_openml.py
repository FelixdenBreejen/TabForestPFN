from typing import Iterator
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from tabularbench.core.enums import DatasetSize, FeatureType, Task
from tabularbench.data.datafile_openml import OpenmlDatafile
from tabularbench.sweeps.paths_and_filenames import PATH_TO_DATA_SPLIT



class OpenMLDataset():

    def __init__(self, openml_dataset_id: int, task: Task, dataset_size: DatasetSize):

        self.openml_dataset_id = openml_dataset_id
        self.task = task
        self.dataset_size = dataset_size

        datafile = OpenmlDatafile(openml_dataset_id, dataset_size)
        X = datafile.x
        y = datafile.y
        categorical_indicator = datafile.categorical_indicator

        self.X, self.y, self.categorical_indicator = self.do_basic_preprocessing(X, y, categorical_indicator)

        train_val_test_indices_all = np.load(PATH_TO_DATA_SPLIT, allow_pickle=True).item()
        self.train_val_test_indices = train_val_test_indices_all[self.openml_dataset_id][int(self.dataset_size)]

        self.n_splits = len(self.train_val_test_indices)


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
        

        for idcs in self.train_val_test_indices:
            
            train_idcs = idcs['train']
            val_idcs = idcs['val']
            test_idcs = idcs['test']

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



