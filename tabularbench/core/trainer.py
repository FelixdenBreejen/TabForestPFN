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


    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):





        for epoch in range(self.cfg['max_epochs']):
            
            self.model.train()
        
            epoch_statistics_train = EpochStatistics()

            for batch in loader_train:

                x, y = batch
                x = x.to(self.cfg['device'])
                y = y.to(self.cfg['device'])
                y_hat_train = self.model(x)
                loss = self.loss(y_hat_train, y)
                score = self.score(y_hat_train, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_statistics_train.update(loss.item(), score, x.shape[0])

            loss_train, score_train = epoch_statistics_train.get()

            self.model.eval()

            epoch_statistics_valid = EpochStatistics()
            
            with torch.no_grad():

                for batch in loader_valid:
                    x, y = batch
                    x = x.to(self.cfg['device'])
                    y = y.to(self.cfg['device'])
                    y_hat_valid = self.model(x)
                    loss_valid = self.loss(y_hat_valid, y)
                    score_valid = self.score(y_hat_valid, y)
                    
                    epoch_statistics_valid.update(loss_valid.item(), score_valid, x.shape[0])

            loss_valid, score_valid = epoch_statistics_valid.get()

            print(f"Epoch {epoch} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Valid loss: {loss_valid:.4f} | Valid score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                print("Early stopping")
                break

            self.scheduler.step(loss_valid)


    def predict(self, x: np.ndarray):

        self.model.eval()

        dataset = torch.utils.data.TensorDataset(torch.Tensor(x))
        loader = self.make_loader(dataset, training=False)

        y_hat = []

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self.cfg['device'])
                output = self.model(x)
                output = output.cpu().numpy()

                if self.cfg['regression']:
                    y_hat.append(output)
                else:
                    y_hat.append(output.argmax(axis=1))

        return np.concatenate(y_hat)    


    def score(self, y_hat, y):

        with torch.no_grad():
            if self.cfg['regression']:  
                return np.sqrt(np.mean((y_hat.cpu().numpy() - y.cpu().numpy())**2))
            else:
                return np.mean((y_hat.cpu().numpy().argmax(axis=1) == y.cpu().numpy()))
            

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

        if self.cfg.task == Task.REGRESSION:
            return (
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x),
                    torch.FloatTensor(y)
               )
            )
        elif self.cfg.task == Task.CLASSIFICATION:
            return (
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x),
                    torch.LongTensor(y)
                )
            )
        

    def make_loader(self, dataset, training):

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.hyperparams.batch_size,
            shuffle=training,
            pin_memory=True,
            drop_last=training
        )

    
    def select_loss(self):

        if self.cfg.task == Task.REGRESSION:
            loss = torch.nn.MSELoss()
        elif self.cfg.task == Task.CLASSIFICATION:
            loss = torch.nn.CrossEntropyLoss()

        return loss


def extract_module_config(model_config, input_shape_config):

    module_config = {}
    total_config = {**model_config, **input_shape_config}

    for key in total_config.keys():
        if key.startswith("module__"):
            module_config[key[len("module__"):]] = total_config[key]

    module_config['regression'] = total_config['regression']
    module_config['categorical_indicator'] = total_config['categorical_indicator']

    return module_config