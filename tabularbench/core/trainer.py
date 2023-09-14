from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np

from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics
from tabularbench.core.enums import Task
from tabularbench.sweeps.run_config import RunConfig


class Trainer(BaseEstimator):

    def __init__(
            self, 
            cfg: RunConfig,
            model: torch.nn.Module
        ) -> None:

        self.cfg = cfg
        self.model = model
        self.model.to(self.cfg.device)
        
        self.loss = self.select_loss()
        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        self.early_stopping = EarlyStopping(patience=self.cfg.hyperparams.early_stopping_patience)
        self.checkpoint = Checkpoint("temp_weights", id=str(self.cfg.device))


    def train(self, x_train: np.ndarray, x_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):

        dataset_train = self.make_dataset(x_train, y_train)
        dataset_valid = self.make_dataset(x_val, y_val)

        dataloader_train = self.make_loader(dataset_train, training=True)
        dataloader_valid = self.make_loader(dataset_valid, training=False)

        for epoch in range(self.cfg.hyperparams.max_epochs):
            loss_train, score_train = self.run_epoch(dataloader_train, training=True)
            loss_valid, score_valid = self.run_epoch(dataloader_valid, training=False)

            self.cfg.logger.info(f"Epoch {epoch} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Valid loss: {loss_valid:.4f} | Valid score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                self.cfg.logger.info("Early stopping")
                break

            self.scheduler.step(loss_valid)


    def test(self, x_test: np.ndarray, y_test: np.ndarray):

        dataset_test = self.make_dataset(x_test, y_test)
        dataloader_test = self.make_loader(dataset_test, training=False)

        _, score_test = self.run_epoch(dataloader_test, training=False)

        return score_test

    
    def run_epoch(self, dataloader: torch.utils.data.DataLoader, training: bool):

        if training:
            self.model.train()
        else:
            self.model.eval()
        
        epoch_statistics = EpochStatistics()

        with torch.set_grad_enabled(training):

            for batch in dataloader:
                x, y = batch
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                y_hat = self.model(x)
                loss = self.loss(y_hat, y)
                score = self.score(y_hat, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_statistics.update(loss.item(), score, x.shape[0])

        loss, score = epoch_statistics.get()
        return loss, score


    def predict(self, x: np.ndarray) -> np.ndarray:

        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x))
        dataloader = self.make_loader(dataset, training=False)

        self.model.eval()

        y_hats = []

        with torch.no_grad():

            for batch in dataloader:
                x = batch[0]
                x = x.to(self.cfg.device)
                y_hat = self.model(x)
                y_hats.append(y_hat.cpu().numpy())

        y_hats = np.concatenate(y_hats, axis=0)
        return y_hats


    def score(self, y_hat, y):

        with torch.no_grad():
            y_hat = y_hat.cpu().numpy()
            y = y.cpu().numpy()

            match self.cfg.task:
                case Task.REGRESSION:                
                    ss_res = np.sum((y - y_hat) ** 2, axis=0)
                    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                    return r2
                case Task.CLASSIFICATION:
                    return np.mean((y_hat.argmax(axis=1) == y))
            

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))


    def select_optimizer(self):

        if self.cfg.hyperparams.optimizer == "adam":
            optimizer = Adam(
                self.model.parameters(), 
                lr=self.cfg.hyperparams.lr,
                betas=(0.9, 0.99),
                weight_decay=self.cfg.hyperparams.weight_decay
            )
        elif self.cfg.hyperparams.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.cfg.hyperparams.lr,
                betas=(0.9, 0.99),
                weight_decay=self.cfg.hyperparams.weight_decay
            )
        elif self.cfg.hyperparams.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg.hyperparams.lr,
                weight_decay=self.cfg.hyperparams.weight_decay
            )
        else:
            raise ValueError("Optimizer not recognized")
        
        return optimizer
        

    def select_scheduler(self):

        if self.cfg.hyperparams.lr_scheduler:      
            scheduler = ReduceLROnPlateau(
                self.optimizer, 
                patience=self.cfg.hyperparams.lr_scheduler_patience, 
                min_lr=2e-5, 
                factor=0.2
            )
        else:
            scheduler = LambdaLR(
                self.optimizer,
                lambda _: 1
            )

        return scheduler
    

    def make_dataset(self, x, y):

        match self.cfg.task:
            case Task.REGRESSION:
                return (
                    torch.utils.data.TensorDataset(
                        torch.FloatTensor(x),
                        torch.FloatTensor(y)
                )
                )
            case Task.CLASSIFICATION:
                return (
                    torch.utils.data.TensorDataset(
                        torch.FloatTensor(x),
                        torch.LongTensor(y)
                    )
                )
        

    def make_loader(self, dataset, training):

        drop_last = training and self.cfg.hyperparams.batch_size > len(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.hyperparams.batch_size,
            shuffle=training,
            pin_memory=True,
            drop_last=drop_last
        )

    
    def select_loss(self):

        match self.cfg.task:
            case Task.REGRESSION:
                return torch.nn.MSELoss()
            case Task.CLASSIFICATION:
                return torch.nn.CrossEntropyLoss()
