from typing import Generator, Iterator

import numpy as np
import torch

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.enums import GeneratorName
from tabularbench.data.preprocessor import Preprocessor
from tabularbench.data.synthetic_generator_selector import SyntheticDatasetGeneratorSelectorMixin


class SyntheticDataset(torch.utils.data.IterableDataset, SyntheticDatasetGeneratorSelectorMixin):

    def __init__(
        self, 
        cfg: ConfigPretrain,
        generator_name: GeneratorName,
        min_samples_support: int,
        max_samples_support: int,
        n_samples_query: int,
        min_features: int,
        max_features: int,
        max_classes: int,
        use_quantile_transformer: bool,
        use_feature_count_scaling: bool,
        generator_hyperparams: dict
    ) -> None:
        
        self.cfg = cfg
        self.generator_name = generator_name
        self.min_samples_support = min_samples_support
        self.max_samples_support = max_samples_support
        self.n_samples_query = n_samples_query
        self.n_samples = max_samples_support + n_samples_query
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.use_quantile_transformer = use_quantile_transformer
        self.use_feature_count_scaling = use_feature_count_scaling
        self.generator_hyperparams = generator_hyperparams


    def __iter__(self) -> Iterator:

        self.synthetic_dataset_generator = self.select_synthetic_dataset_generator()
        return self.generator()
    

    def generator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        
        while True:
            x, y = next(self.synthetic_dataset_generator)

            y = self.randomize_class_order(y)
            
            x_support, y_support, x_query, y_query = self.split_into_support_and_query(x, y)

            preprocessor = Preprocessor(
                max_features=self.max_features,
                use_quantile_transformer=self.use_quantile_transformer,
                use_feature_count_scaling=self.use_feature_count_scaling,
            )

            preprocessor.fit(x_support, y_support)
            x_support = preprocessor.transform(x_support)
            x_query = preprocessor.transform(x_query)

            x_support, x_query = self.randomize_feature_order(x_support, x_query)
            
            x_support = torch.tensor(x_support, dtype=torch.float32)
            y_support = torch.tensor(y_support, dtype=torch.float32)
            x_query = torch.tensor(x_query, dtype=torch.float32)
            y_query = torch.tensor(y_query, dtype=torch.int64)

            yield {
                'x_support': x_support,
                'y_support': y_support,
                'x_query': x_query,
                'y_query': y_query
            }


    def randomize_class_order(self, y):
            
        curr_classes = int(y.max().item()) + 1
        new_classes = np.random.permutation(self.max_classes)
        mapping = { i: new_classes[i] for i in range(curr_classes) }
        y = np.array([mapping[i.item()] for i in y], dtype=np.int64)

        return y


    def split_into_support_and_query(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        curr_samples = x.shape[0]

        n_samples_support = np.random.randint(low=self.min_samples_support, high=self.max_samples_support)
        rand_index = np.random.permutation(curr_samples)

        rand_support_index = rand_index[:n_samples_support]
        rand_query_index = rand_index[n_samples_support:n_samples_support+self.n_samples_query]

        x_support = x[rand_support_index]
        y_support = y[rand_support_index]
        x_query = x[rand_query_index]
        y_query = y[rand_query_index]

        return x_support, y_support, x_query, y_query
    

    def randomize_feature_order(self, x_support: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        curr_features = x_support.shape[1]
        new_feature_order = torch.randperm(curr_features)

        x_support = x_support[:, new_feature_order]
        x_query = x_query[:, new_feature_order]

        return x_support, x_query
    
