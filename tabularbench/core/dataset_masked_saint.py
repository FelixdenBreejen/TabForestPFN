import torch
import numpy as np
from sklearn.model_selection import train_test_split



class MaskedSaintDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray = None,
        batch_size: int = 1024,
        test_prop: float = 0.2,
    ):
        """
        test_prop: proportion of the test data to use for the query
        if y_test is None, we use a proxy of class = -1
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.n_train_observations = x_train.shape[0]
        self.n_test_observations = x_test.shape[0]

        if self.y_test is None:
            self.y_test = np.zeros((self.n_test_observations,), dtype=np.int64) - 1

        batch_size_test = int(batch_size * test_prop)
        batch_size_train = batch_size - batch_size_test

        self.batch_size_test = min(batch_size_test, self.n_test_observations)
        self.batch_size_train = min(batch_size_train, self.n_train_observations)

        mean, std = self.calc_mean_std(x_train)
        self.x_train = self.normalize_by_mean_std(self.x_train, mean, std)
        self.x_test = self.normalize_by_mean_std(self.x_test, mean, std)

        self.x_tests = self.split_in_chunks(self.x_test, self.batch_size_test)
        self.y_tests = self.split_in_chunks(self.y_test, self.batch_size_test)


    def __len__(self):
        return len(self.x_tests)

    def __getitem__(self, idx):

        train_indices = np.random.choice(
            self.n_train_observations, 
            size=self.batch_size_train, 
            replace=False
        )

        x_train = torch.tensor(self.x_train[train_indices], dtype=torch.float)
        y_train = torch.tensor(self.y_train[train_indices], dtype=torch.long)
        x_test = torch.tensor(self.x_tests[idx], dtype=torch.float)
        y_test = torch.tensor(self.y_tests[idx], dtype=torch.long)

        x_both = torch.concatenate([x_train, x_test], axis=0)
        y_both = torch.concatenate([y_train, y_test], axis=0)

        x_size_mask = torch.zeros_like(x_both, dtype=torch.bool)
        y_size_mask = torch.zeros_like(y_both, dtype=torch.bool)
        y_label_mask = torch.concatenate([
            torch.zeros_like(y_train, dtype=torch.bool), 
            torch.ones_like(y_test, dtype=torch.bool)
        ], axis=0)

        return x_both, x_size_mask, y_both, y_size_mask, y_label_mask
        

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

        # in case of a unique feature: don't divide by zero
        std[std == 0] = 1
        x = (x - mean) / std

        assert np.isnan(x).sum() == 0

        return x


    def split_in_chunks(self, x: np.ndarray, batch_size: int) -> list[np.ndarray]:
        """
        Splits the data into chunks of size batch_size
        """

        n_chunks = int(np.ceil(x.shape[0] / batch_size))
        x_chunks = []

        for i in range(n_chunks):
            x_chunks.append(x[i * batch_size: (i + 1) * batch_size])

        return x_chunks
    





def MaskedSaintDatasetGenerator(
    x: np.ndarray, 
    y: np.ndarray, 
    batch_size: int = 1024,
    split: float = 0.8,
):
        
    while True:

        x_train, x_test, y_train, y_test = train_test_split(
            x, 
            y, 
            train_size=split, 
            shuffle=True,
            stratify=y
        )

        static_dataset = MaskedSaintDataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            batch_size=batch_size
        )

        yield static_dataset



    

        