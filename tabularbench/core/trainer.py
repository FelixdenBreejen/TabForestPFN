from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
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
            model: torch.nn.Module,
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

        self.y_transformer = self.create_y_transformer(y_train)

        dataset_train = self.make_dataset_xy(x_train, self.y_transformer.transform(y_train))
        dataset_valid = self.make_dataset_x(x_val)

        dataloader_train = self.make_loader(dataset_train, training=True)
        dataloader_valid = self.make_loader(dataset_valid, training=False)

        for epoch in range(self.cfg.hyperparams.max_epochs):
            loss_train, score_train = self.train_epoch(dataloader_train)
            loss_valid, score_valid = self.test_epoch(dataloader_valid, y_val)

            self.cfg.logger.info(f"Epoch {epoch:03d} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Val loss: {loss_valid:.4f} | Val score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                self.cfg.logger.info("Early stopping")
                break

            self.scheduler.step(loss_valid)


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
    

    def test(self, x_test: np.ndarray, y_test: np.ndarray):

        self.load_params(self.checkpoint.path)

        dataset_test = self.make_dataset_x(x_test)
        dataloader_test = self.make_loader(dataset_test, training=False)
        loss_test, score_test = self.test_epoch(dataloader_test, y_test)

        return loss_test, score_test


    def test_epoch(self, dataloader_x: torch.utils.data.DataLoader, y_test: np.ndarray):
        
        y_hats = self.predict_epoch(dataloader_x)
        y_hats = self.y_transformer.inverse_transform(y_hats)

        loss_test = self.loss(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()
        score_test = self.score(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()
        return loss_test, score_test
    

    def predict(self, x: np.ndarray):

        dataset_x = self.make_dataset_x(x)
        dataloader_x = self.make_loader(dataset_x, training=False)
        y_hats = self.predict_epoch(dataloader_x)
        y_hats = self.y_transformer.inverse_transform(y_hats)

        return y_hats


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
                ss_tot = torch.sum((y - torch.mean(y)) ** 2, axis=0)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                return r2
            case Task.CLASSIFICATION:
                return (y_hat.argmax(axis=1) == y).sum() / len(y)
            

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))


    def create_y_transformer(self, y_train: np.ndarray) -> TransformerMixin:
        # The y_transformer transformers the target variable to a normal distribution
        # This should be used for the y variable when training a regression model,
        # but when testing the model, we want to inverse transform the predictions

        match self.cfg.task:
            case Task.REGRESSION:
                y_transformer = QuantileTransformer1D(output_distribution="normal")
                y_transformer.fit(y_train)
                return y_transformer
            case Task.CLASSIFICATION:
                return FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)


    def select_optimizer(self):

        if self.cfg.hyperparams.optimizer == "adam":
            optimizer = Adam(
                self.model.parameters(), 
                lr=self.cfg.hyperparams.lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.hyperparams.weight_decay
            )
        elif self.cfg.hyperparams.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.cfg.hyperparams.lr,
                betas=(0.9, 0.999),
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




class QuantileTransformer1D(BaseEstimator, TransformerMixin):

    def __init__(self, output_distribution="normal") -> None:
        self.quantile_transformer = QuantileTransformer(output_distribution=output_distribution)

    def fit(self, x: np.ndarray):
        self.quantile_transformer.fit(x[:, None])
        return self
    
    def transform(self, x: np.ndarray):
        return self.quantile_transformer.transform(x[:, None])[:, 0]
    
    def inverse_transform(self, x: np.ndarray):
        return self.quantile_transformer.inverse_transform(x[:, None])[:, 0]