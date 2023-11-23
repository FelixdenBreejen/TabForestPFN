from typing import Generator, Iterator

import torch

from tabularbench.models.tabPFN.synthetic_data import synthetic_dataset_generator



class SyntheticDataset(torch.utils.data.IterableDataset):

    def __init__(
        self, 
        min_samples: int,
        max_samples: int,
        min_features: int,
        max_features: int,
        max_classes: int,
        support_prop: float = 0.8
    ) -> None:
        
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.support_prop = support_prop
        
        self.synthetic_dataset_generator = synthetic_dataset_generator(
            min_samples=min_samples,
            max_samples=max_samples,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes
        )


    def __iter__(self) -> Iterator:
        return self.generator()
    

    def generator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        
        while True:
            x, y = next(self.synthetic_dataset_generator)

            assert torch.all(y >= 0)
            assert torch.all(torch.isfinite(x))

            y = self.randomize_classes(y)
            x_support, y_support, x_query, y_query = self.split_into_support_and_query(x, y)
            y_support = self.scale_y_support(y_support)
            x_support, x_query = self.normalize_features(x_support, x_query)
            x_support, x_query = self.expand_feature_dimension_to_max_features(x_support, x_query)
            
            assert torch.all(torch.isfinite(x))

            yield {
                'x_support': x_support,
                'y_support': y_support,
                'x_query': x_query,
                'y_query': y_query
            }


    def randomize_classes(self, y):
            
        curr_classes = int(y.max().item()) + 1
        new_classes = torch.randperm(self.max_classes)
        mapping = { i: new_classes[i] for i in range(curr_classes) }
        y = torch.tensor([mapping[i.item()] for i in y], dtype=torch.long)

        return y
    

    def scale_y_support(self, y_support: torch.Tensor) -> torch.Tensor:

        y_support = y_support.float()
        y_support = y_support / self.max_classes
        
        return y_support


    def split_into_support_and_query(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        curr_samples = x.shape[0]

        curr_samples_support = int(curr_samples*self.support_prop)

        rand_index = torch.randperm(curr_samples)
        rand_support_index = rand_index[:curr_samples_support]
        rand_query_index = rand_index[curr_samples_support:]

        x_support = x[rand_support_index]
        y_support = y[rand_support_index]
        x_query = x[rand_query_index]
        y_query = y[rand_query_index]

        return x_support, y_support, x_query, y_query
    

    def normalize_features(self, x_support: torch.Tensor, x_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x_mean = x_support.mean(dim=0)
        x_std = x_support.std(dim=0)

        x_support = (x_support - x_mean[None, :]) / x_std[None, :]
        x_query = (x_query - x_mean[None, :]) / x_std[None, :]

        # for singular values (features with one unique value), set it to zero
        x_support[:, x_std == 0] = 0
        x_query[:, x_std == 0] = 0

        return x_support, x_query
    

    def expand_feature_dimension_to_max_features(self, x_support: torch.Tensor, x_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        curr_features = x_support.shape[1]
        curr_samples_support = x_support.shape[0]
        curr_samples_query = x_query.shape[0]

        x_support_expanded = torch.zeros((curr_samples_support, self.max_features), dtype=torch.float)
        x_support_expanded[:, :curr_features] = x_support

        x_query_expanded = torch.zeros((curr_samples_query, self.max_features), dtype=torch.float)
        x_query_expanded[:, :curr_features] = x_query

        return x_support_expanded, x_query_expanded
    
