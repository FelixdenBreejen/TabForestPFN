from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabularbench.config.config_run import ConfigRun
from tabularbench.core.callbacks import Checkpoint, EarlyStopping, EpochStatistics
from tabularbench.core.enums import Task
from tabularbench.core.get_loss import get_loss
from tabularbench.core.get_optimizer import get_optimizer
from tabularbench.core.get_scheduler import get_scheduler
from tabularbench.core.y_transformer import create_y_transformer


class Trainer(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigRun,
            model: torch.nn.Module,
            n_classes: int
        ) -> None:

        self.cfg = cfg
        self.model = model
        self.model.to(self.cfg.device)
        self.n_classes = n_classes
        
        self.loss = get_loss(self.cfg.task)
        self.optimizer = get_optimizer(self.cfg.hyperparams, self.model)
        self.scheduler = get_scheduler(self.cfg.hyperparams, self.optimizer)

        self.early_stopping = EarlyStopping(patience=self.cfg.hyperparams['early_stopping_patience'])
        self.checkpoint = Checkpoint(Path("temp_weights"), id=str(self.cfg.device))


    def train(self, x_train: np.ndarray, y_train: np.ndarray):

        self.y_transformer = create_y_transformer(y_train, self.cfg.task)

        x_train_train, x_train_val, y_train_train, y_train_val = self.make_train_split(x_train, y_train)

        dataset_train = self.make_dataset_xy(x_train_train, self.y_transformer.transform(y_train_train))
        dataset_valid = self.make_dataset_x(x_train_val)

        dataloader_train = self.make_loader(dataset_train, training=True)
        dataloader_valid = self.make_loader(dataset_valid, training=False)

        for epoch in range(self.cfg.hyperparams['max_epochs']):
            loss_train, score_train = self.train_epoch(dataloader_train)
            loss_valid, score_valid = self.test_epoch(dataloader_valid, y_train_val)

            logger.info(f"Epoch {epoch:03d} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Val loss: {loss_valid:.4f} | Val score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                logger.info("Early stopping")
                break

            self.scheduler.step(loss_valid)
    

    def test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        # x_train and y_train are not used
        # they are passed to be consistent with the API of the other trainers

        self.load_params(self.checkpoint.path)

        y_hats = self.predict(x_test)

        loss_test = self.loss(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()
        score_test = self.score(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()

        return loss_test, score_test
    

    def test_epoch(self, dataloader: torch.utils.data.DataLoader, y_test: np.ndarray):

        y_hats = self.predict_epoch(dataloader)
        
        loss_test = self.loss(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()
        score_test = self.score(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()

        return loss_test, score_test
    

    def predict(self, x: np.ndarray):

        dataset_x = self.make_dataset_x(x)
        dataloader_x = self.make_loader(dataset_x, training=False)
        y_hats = self.predict_epoch(dataloader_x)
        y_hats = self.y_transformer.inverse_transform(y_hats)

        return y_hats


    def train_epoch(self, dataloader: torch.utils.data.DataLoader):

        self.model.train()
        
        epoch_statistics = EpochStatistics()

        for batch in dataloader:
            x, y = batch

            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            y_hat = self.model(x)

            if self.cfg.task == Task.REGRESSION:
                y_hat = torch.squeeze(y_hat)

            loss = self.loss(y_hat, y)
            score = self.score(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_statistics.update(loss.item(), score, x.shape[0])

        loss, score = epoch_statistics.get()
        return loss, score


    def predict_epoch(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:

        self.model.eval()
        y_hats = []

        with torch.no_grad():

            for batch in dataloader:
                x = batch[0]
                x = x.to(self.cfg.device)
                y_hat = self.model(x)

                if self.cfg.task == Task.REGRESSION:
                    y_hat = torch.squeeze(y_hat)
                    
                y_hats.append(y_hat.cpu().numpy())

        y_hats_arr = np.concatenate(y_hats, axis=0)
        return y_hats_arr



    def score(self, y_hat, y):

        match self.cfg.task:
            case Task.REGRESSION:                
                ss_res = torch.sum((y - y_hat) ** 2)
                ss_tot = torch.sum((y - torch.mean(y)) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                return r2
            case Task.CLASSIFICATION:
                return (y_hat.argmax(axis=1) == y).sum() / len(y)
            

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))


    def make_train_split(self, x_train, y_train):

        match self.cfg.task:
            case Task.REGRESSION:
                x_t_train, x_t_valid, y_t_train, y_t_valid = train_test_split(
                    x_train, y_train, test_size=0.2
                )
            case Task.CLASSIFICATION:
                skf = StratifiedKFold(n_splits=5, shuffle=True)
                indices = next(skf.split(x_train, y_train))
                x_t_train, x_t_valid = x_train[indices[0]], x_train[indices[1]]
                y_t_train, y_t_valid = y_train[indices[0]], y_train[indices[1]]

        return x_t_train, x_t_valid, y_t_train, y_t_valid
    

    def make_dataset_xy(self, x, y):
        return (
            torch.utils.data.TensorDataset(
                torch.as_tensor(x),
                torch.as_tensor(y)
            )
        )
            

    def make_dataset_x(self, x):   
        return torch.utils.data.TensorDataset(
            torch.as_tensor(x)
        )
        

    def make_loader(self, dataset, training):

        drop_last = training and self.cfg.hyperparams['batch_size'] > len(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.hyperparams['batch_size'],
            shuffle=training,
            pin_memory=True,
            drop_last=drop_last
        )

