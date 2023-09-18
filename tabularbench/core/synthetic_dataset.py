from typing import Iterator

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
        mask_prop: float = 0.2
    ) -> None:
        
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.mask_prop = mask_prop
        
        self.generator = synthetic_dataset_generator(
            min_samples=min_samples,
            max_samples=max_samples,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_classes
        )


    def __iter__(self) -> Iterator:
        return self.make_dataset()
    

    def make_dataset(self):
        
        while True:
            x, y = next(self.generator)

            assert torch.all(y >= 0)
            assert torch.all(torch.isfinite(x))

            y = self.randomize_classes(y)
            y, y_label_mask = self.randomly_mask_labels(y)
            x = self.normalize_features(x, y_label_mask)
            x, x_size_mask, y, y_size_mask, y_label_mask = self.expand_dimension_to_max_samples_and_features(x, y, y_label_mask)
            
            assert torch.all(torch.isfinite(x))

            yield x, x_size_mask, y, y_size_mask, y_label_mask


    def randomize_classes(self, y):
            
        curr_classes = int(y.max().item()) + 1
        new_classes = torch.randperm(self.max_classes)
        mapping = { i: new_classes[i] for i in range(curr_classes) }
        y = torch.tensor([mapping[i.item()] for i in y], dtype=torch.long)

        return y


    def randomly_mask_labels(self, y):
        """
        Masking means: we ignore the value
        """

        curr_samples = y.shape[0]

        y_label_mask = torch.zeros(curr_samples, dtype=torch.bool)
        n_samples_masked = int(curr_samples*self.mask_prop)
        indices_masked = torch.randperm(curr_samples)[0:n_samples_masked]
        y_label_mask[indices_masked] = 1

        return y, y_label_mask
    

    def normalize_features(self, x, y_label_mask):

        x_not_masked = x[~y_label_mask]

        x_mean = x_not_masked.mean(dim=0)
        x_std = x_not_masked.std(dim=0)

        x = (x - x_mean[None, :]) / x_std[None, :]

        # for singular values, set it to zero instead of removing them
        x[:, x_std == 0] = 0

        return x
    

    def expand_dimension_to_max_samples_and_features(self, x_tight, y_tight, y_label_mask_tight):
            
        curr_samples = x_tight.shape[0]
        curr_features = x_tight.shape[1]

        x_size_mask = torch.ones(self.max_samples, self.max_features, dtype=torch.bool)
        x_size_mask[:curr_samples, :curr_features] = 0

        x = torch.zeros(self.max_samples, self.max_features)
        x[:curr_samples, :curr_features] = x_tight

        y_size_mask = torch.ones(self.max_samples, dtype=torch.bool)
        y_size_mask[:curr_samples] = 0

        y = torch.zeros(self.max_samples, dtype=torch.long) - 1
        y[:curr_samples] = y_tight

        y_label_mask = torch.zeros(self.max_samples, dtype=torch.bool)
        y_label_mask[:curr_samples] = y_label_mask_tight

        return x, x_size_mask, y, y_size_mask, y_label_mask
    



    

        