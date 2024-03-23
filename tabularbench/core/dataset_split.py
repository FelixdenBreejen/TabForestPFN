import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabularbench.core.enums import Task


def make_dataset_split(x: np.ndarray, y: np.ndarray, task: Task) -> tuple[np.ndarray, ...]:
    # Splits the dataset into train and validation sets with ratio 80/20

    size_of_smallest_class = np.min(np.bincount(y))

    if task == Task.CLASSIFICATION and size_of_smallest_class >= 5:
        # stratification needs have at least 5 samples in each class if split is 80/20
        return make_stratified_dataset_split(x, y)
    else:
        return make_standard_dataset_split(x, y)
    

def make_stratified_dataset_split(x, y):

    # Stratify doesn't shuffle the data, so we shuffle it first
    permutation = np.random.permutation(len(y))
    x, y = x[permutation], y[permutation]

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    indices = next(skf.split(x, y))
    x_t_train, x_t_valid = x[indices[0]], x[indices[1]]
    y_t_train, y_t_valid = y[indices[0]], y[indices[1]]

    return x_t_train, x_t_valid, y_t_train, y_t_valid


def make_standard_dataset_split(x, y):
        
    return train_test_split(
        x, y, test_size=0.2
    )