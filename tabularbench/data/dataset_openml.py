from typing import Iterator
import numpy as np
import openml
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from tabularbench.core.enums import DatasetSize, FeatureType, Task
from tabularbench.sweeps.paths_and_filenames import PATH_TO_DATA_SPLIT



class OpenMLDataset():

    def __init__(self, openml_dataset_id: int, task: Task, feature_type: FeatureType, dataset_size: DatasetSize):
        self.openml_dataset_id = openml_dataset_id
        self.task = task
        self.feature_type = feature_type
        self.dataset_size = dataset_size

        dataset = openml.datasets.get_dataset(self.openml_dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)
        
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )
        X = X.to_numpy()
        y = y.to_numpy()

        self.X, self.y, self.categorical_indicator = self.do_basic_preprocessing(X, y, categorical_indicator)

        train_val_test_indices_all = np.load(PATH_TO_DATA_SPLIT, allow_pickle=True).item()
        self.train_val_test_indices = train_val_test_indices_all[self.openml_dataset_id][int(self.dataset_size)]

        self.n_splits = len(self.train_val_test_indices)


    def do_basic_preprocessing(self, X, y, categorical_indicator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if y.dtype == np.dtype('O'):
            y = LabelEncoder().fit_transform(y)

        match self.feature_type:
            case FeatureType.NUMERICAL:
                assert categorical_indicator is None or not np.array(categorical_indicator).astype(bool).any(), "There are categorical features in the dataset"
                categorical_indicator = np.zeros(X.shape[1]).astype(bool)
            case FeatureType.CATEGORICAL:
                assert categorical_indicator is None or np.array(categorical_indicator).astype(bool).all(), "There are numerical features in the dataset"
                categorical_indicator = np.ones(X.shape[1]).astype(bool)
            case FeatureType.MIXED:
                assert categorical_indicator is not None, "There is no information about the feature types in the dataset"
                assert np.array(categorical_indicator).astype(bool).any(), "There are no categorical features in the dataset"
                assert not np.array(categorical_indicator).astype(bool).all(), "There are no numerical features in the dataset"
                categorical_indicator = np.array(categorical_indicator).astype(bool)

        match self.task:
            case Task.CLASSIFICATION:
                y = y.astype(np.int64)
            case Task.REGRESSION:
                y = y.astype(np.float32)

        X = X.astype(np.float32)

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



