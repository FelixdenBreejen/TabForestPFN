import random

from tqdm import tqdm

from tabularbench.data.synthetic_generator_forest import synthetic_dataset_generator_forest
from tabularbench.data.synthetic_generator_tabpfn import synthetic_dataset_generator_tabpfn


def synthetic_dataset_generator_mix(
    min_features = 3,
    max_features = 100,
    n_samples = 10000,
    max_classes = 10,
    **forest_kwargs
):

    generator_tabpfn = synthetic_dataset_generator_tabpfn(
        min_features=min_features, 
        max_features=max_features, 
        n_samples=n_samples, 
        max_classes=max_classes
    )

    generator_forest = synthetic_dataset_generator_forest(
        min_features=min_features, 
        max_features=max_features, 
        n_samples=n_samples, 
        max_classes=max_classes, 
        **forest_kwargs
    )

    while True:

        if random.random() < 0.5:
            yield next(generator_forest)
        else:
            yield next(generator_tabpfn)



if __name__ == '__main__':

    generator = synthetic_dataset_generator_mix(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        base_size = 10000,
        n_estimators = 1,
        min_depth = 2,
        max_depth = 2,
    )

    for _ in tqdm(range(1)):        
        x, y = next(generator)