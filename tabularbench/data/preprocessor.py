from loguru import logger
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer


class Preprocessor(TransformerMixin, BaseEstimator):
    """
    This class is used to preprocess the data before it is pushed through the model.
    The preprocessor assures that the data has the right shape and is normalized,
    This way the model always gets the same input distribution, 
    no matter whether the input data is synthetic or real.

    """

    def __init__(
            self, 
            max_features: int,
            use_quantile_transformer: bool,
            use_feature_count_scaling: bool,
        ):

        self.max_features = max_features
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling

    
    def fit(self, X: np.ndarray, y: np.ndarray = None):

        X = self.cutoff_excess_features(X, self.max_features)
        
        self.singular_features = X.std(axis=0) == 0
        X = self.cutoff_singular_features(X, self.singular_features)

        if self.use_quantile_transformer:
            n_quantiles = min(X.shape[0], 1000)
            self.quantile_transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
            X = self.quantile_transformer.fit_transform(X)
        
        self.mean, self.std = self.calc_mean_std(X)

        return self
    

    def transform(self, X: np.ndarray, y: np.ndarray = None):

        X = self.cutoff_excess_features(X, self.max_features)
        X = self.cutoff_singular_features(X, self.singular_features)

        if self.use_quantile_transformer:
            X = self.quantile_transformer.transform(X)
        
        X = self.normalize_by_mean_std(X, self.mean, self.std)

        if self.use_feature_count_scaling:
            X = self.normalize_by_feature_count(X, self.max_features)

        X = self.extend_features(X, self.max_features)

        if y is None:
            return X

        return X, y
        

    def cutoff_excess_features(self, x: np.ndarray, max_features: int) -> np.ndarray:

        if x.shape[1] > max_features:
            logger.info(f"TabPFN allows {max_features} features, but the dataset has {x.shape[1]} features. Excess features are cut off.")
            x = x[:, :max_features]

        return x
    

    def cutoff_singular_features(self, x: np.ndarray, singular_features: np.ndarray) -> np.ndarray:

        if singular_features.any():
            x = x[:, ~singular_features]

        return x


    def calc_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and std of the training data
        """
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        return mean, std
    

    def normalize_by_mean_std(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalizes the data by the mean and std
        """

        x = (x - mean) / std
        return x


    def normalize_by_feature_count(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        An interesting way of normalization by the tabPFN paper
        """

        x = x * max_features / x.shape[1]
        return x



    def extend_features(self, x: np.ndarray, max_features) -> np.ndarray:
        """
        Increases the number of features to the number of features the tab pfn model has been trained on
        """
        added_zeros = np.zeros((x.shape[0], max_features - x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, added_zeros], axis=1)
        return x

    
