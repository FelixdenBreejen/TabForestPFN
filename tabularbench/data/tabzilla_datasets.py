# this script is based on the TabSurvey script and function load_data, which returns a specific dataset (X, y).
# instead, this script "prepares" each dataset implemented in our codebase, by doing the following:
# - reading the dataset from a file or online source
# - applying any necessary cleaning (no pre-processing, like encoding or scaling variables)
# - writes each dataset to its own local directory. each dataset directory will contain:
# -- a compressed version of the dataset (X.npy and y.npy)
# -- a json containing metadata


import gzip
import json
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from sklearn.preprocessing import LabelEncoder


class TabularDataset(object):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: Optional[int] = None,
        num_instances: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indeces: Optional[list] = None,
        split_source: Optional[str] = None,
    ) -> None:
        """
        name: name of the dataset
        X: matrix of shape (num_instances x num_features)
        y: array of length (num_instances)
        cat_idx: indices of categorical features
        target_type: {"regression", "classification", "binary"}
        num_classes: 1 for regression 2 for binary, and >2 for classification
        num_features: number of features
        num_instances: number of instances
        split_indeces: specifies dataset splits as a list of dictionaries, with entries "train", "val", and "test".
            each entry specifies the indeces corresponding to the train, validation, and test set.
        """
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must match along their 0-th dimensions"
        assert len(X.shape) == 2, "X must be 2-dimensional"
        assert len(y.shape) == 1, "y must be 1-dimensional"

        if num_instances is not None:
            assert (
                X.shape[0] == num_instances
            ), f"first dimension of X must be equal to num_instances. X has shape {X.shape}"
            assert y.shape == (
                num_instances,
            ), f"shape of y must be (num_instances, ). y has shape {y.shape} and num_instances={num_instances}"
        else:
            num_instances = X.shape[0]

        if num_features is not None:
            assert (
                X.shape[1] == num_features
            ), f"second dimension of X must be equal to num_features. X has shape {X.shape}"
        else:
            num_features = X.shape[1]

        if len(cat_idx) > 0:
            assert (
                max(cat_idx) <= num_features - 1
            ), f"max index in cat_idx is {max(cat_idx)}, but num_features is {num_features}"
        assert target_type in ["regression", "classification", "binary"]

        if target_type == "regression":
            assert num_classes == 1
        elif target_type == "binary":
            assert num_classes == 1
        elif target_type == "classification":
            assert num_classes > 2

        self.name = name
        self.X = X
        self.y = y
        self.cat_idx = cat_idx
        self.target_type = target_type
        self.num_classes = num_classes
        self.num_features = num_features
        self.cat_dims = cat_dims
        self.num_instances = num_instances
        self.split_indeces = split_indeces
        self.split_source = split_source

        pass

    def target_encode(self):
        # print("target_encode...")
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

        # Sanity check
        if self.target_type == "classification":
            assert self.num_classes == len(
                le.classes_
            ), "num_classes was set incorrectly."

    def cat_feature_encode(self):
        # print("cat_feature_encode...")
        if not self.cat_dims is None:
            raise RuntimeError(
                "cat_dims is already set. Categorical feature encoding might be running twice."
            )
        self.cat_dims = []

        # Preprocess data
        for i in range(self.num_features):
            if self.cat_idx and i in self.cat_idx:
                le = LabelEncoder()
                self.X[:, i] = le.fit_transform(self.X[:, i])

                # Setting this?
                self.cat_dims.append(len(le.classes_))

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "target_type": self.target_type,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": self.num_instances,
            "split_source": self.split_source,
        }

    @classmethod
    def read(cls, p: Path):
        """read a dataset from a folder"""

        # make sure that all required files exist in the directory
        X_path = p.joinpath("X.npy.gz")
        y_path = p.joinpath("y.npy.gz")
        metadata_path = p.joinpath("metadata.json")
        split_indeces_path = p / "split_indeces.npy.gz"

        assert X_path.exists(), f"path to X does not exist: {X_path}"
        assert y_path.exists(), f"path to y does not exist: {y_path}"
        assert (
            metadata_path.exists()
        ), f"path to metadata does not exist: {metadata_path}"
        assert (
            split_indeces_path.exists()
        ), f"path to split indeces does not exist: {split_indeces_path}"

        # read data
        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)
        with gzip.GzipFile(split_indeces_path, "rb") as f:
            split_indeces = np.load(f, allow_pickle=True)

        # read metadata
        with open(metadata_path, "r") as f:
            kwargs = json.load(f)

        kwargs["X"], kwargs["y"], kwargs["split_indeces"] = X, y, split_indeces
        return cls(**kwargs)


    def write(self, path: Path, overwrite=True) -> None:
        """Save the dataset as a netcdf file."""

        name_parts = self.name.split("__")
        openml_dataset_id = int(name_parts[2])
        openml_dataset_name = name_parts[1]

        n_observations = self.X.shape[0]
        n_features = self.X.shape[1]
        n_splits = len(self.split_indeces)

        split_train = np.zeros((n_observations, n_splits), dtype=bool)
        split_val = np.zeros((n_observations, n_splits), dtype=bool)
        split_test = np.zeros((n_observations, n_splits), dtype=bool)

        for i in range(n_splits):
            split_train[self.split_indeces[i]['train'], i] = True
            split_val[self.split_indeces[i]['val'], i] = True
            split_test[self.split_indeces[i]['test'], i] = True

        categorical_indicator = np.zeros(n_features, dtype=bool)
        for i in self.cat_idx:
            categorical_indicator[i] = True

        attribute_names = np.array([f"feature_{i}" for i in range(n_features)])

        X = self.X.astype(np.float32)

        self.ds = xr.Dataset(
            data_vars={
                'x': (['observation', 'feature'], X),
                'y': (['observation'], self.y),
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
                'openml_dataset_id': openml_dataset_id,
                'openml_dataset_name': openml_dataset_name,
            }
        )
    
        self.ds.to_netcdf(path / f"tabzilla_{openml_dataset_id}")


    def write_legacy(self, p: Path, overwrite=False) -> None:
        """original write function made by tabzilla"""

        if not overwrite:
            assert ~p.exists(), f"the path {p} already exists."

        # create the folder
        p.mkdir(parents=True, exist_ok=overwrite)

        # write data
        with gzip.GzipFile(p.joinpath("X.npy.gz"), "w") as f:
            np.save(f, self.X)
        with gzip.GzipFile(p.joinpath("y.npy.gz"), "w") as f:
            np.save(f, self.y)
        with gzip.GzipFile(p.joinpath("split_indeces.npy.gz"), "wb") as f:
            np.save(f, self.split_indeces)

        # write metadata
        with open(p.joinpath("metadata.json"), "w") as f:
            metadata = self.get_metadata()
            json.dump(self.get_metadata(), f, indent=4)
