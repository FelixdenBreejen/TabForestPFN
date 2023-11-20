from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import numpy as np

from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics
from tabularbench.core.enums import Task
from tabularbench.core.get_loss import get_loss
from tabularbench.core.get_optimizer import get_optimizer
from tabularbench.core.get_scheduler import get_scheduler
from tabularbench.core.y_transformer import create_y_transformer
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics
from tabularbench.data.dataset_tabpfn_finetune import TabPFNFinetuneDataset, TabPFNFinetuneGenerator


class Trainer(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigRun,
            model: torch.nn.Module,
        ) -> None:

        self.cfg = cfg
        self.model = model
        self.model.to(self.cfg.device)
        
        self.loss = get_loss(self.cfg.task)
        self.optimizer = get_optimizer(self.cfg.hyperparams, self.model)
        self.scheduler = get_scheduler(self.cfg.hyperparams, self.optimizer)

        self.early_stopping = EarlyStopping(patience=self.cfg.hyperparams.early_stopping_patience)
        self.checkpoint = Checkpoint("temp_weights", id=str(self.cfg.device))


    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        
        self.y_transformer = create_y_transformer(y_train, self.cfg.task)

        a = self.make_dataset_split(x_train=x_train, y_train=y_train)
        x_train_train, x_train_valid, y_train_train, y_train_valid = a

        dataset_train_generator = TabPFNFinetuneGenerator(
            self.cfg,
            x_train_train,
            self.y_transformer.transform(y_train_train),
            regression=self.cfg.task == Task.REGRESSION,
            batch_size=self.cfg.hyperparams.batch_size,
        )

        dataset_valid = TabPFNFinetuneDataset(
            self.cfg,
            x_train_train, 
            self.y_transformer.transform(y_train_train), 
            x_train_valid, 
            y_train_valid,
            regression=self.cfg.task == Task.REGRESSION,
            batch_size=self.cfg.hyperparams.batch_size
        )

        loader_valid = self.make_loader(dataset_valid, training=False)

        for epoch in range(self.cfg.hyperparams.max_epochs):

            dataset_train = next(dataset_train_generator)            
            loader_train = self.make_loader(dataset_train, training=True)
            
            loss_train, score_train = self.train_epoch(loader_train)
            loss_valid, score_valid = self.test_epoch(loader_valid, y_train_valid)

            self.cfg.logger.info(f"Epoch {epoch:03d} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Val loss: {loss_valid:.4f} | Val score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                self.cfg.logger.info("Early stopping")
                break

            self.scheduler.step(loss_valid)

        return self
    
    
    def test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):

        self.load_params(self.checkpoint.path)

        y_hats = self.predict(x_train, self.y_transformer.transform(y_train), x_test)

        loss_test = self.loss(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()
        score_test = self.score(torch.as_tensor(y_hats), torch.as_tensor(y_test)).item()

        return loss_test, score_test
    

    def test_epoch(self, dataloader: torch.utils.data.DataLoader, y_test: np.ndarray):
        
        y_hat = self.predict_epoch(dataloader)  
        y_hat_finish = self.y_transformer.inverse_transform(y_hat)

        loss_test = self.loss(torch.as_tensor(y_hat_finish), torch.as_tensor(y_test)).item()
        score_test = self.score(torch.as_tensor(y_hat_finish), torch.as_tensor(y_test)).item()
        return loss_test, score_test
    

    def predict(self, x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray):

        dataset = TabPFNFinetuneDataset(self.cfg, x_train, y_train, x, batch_size=self.cfg.hyperparams.batch_size)
        loader = self.make_loader(dataset, training=False)

        y_hat_list = []

        for _ in range(self.cfg.hyperparams.n_ensembles):
            y_hat = self.predict_epoch(loader)
            y_hat_list.append(y_hat)

        y_hat_ensembled = sum(y_hat_list) / len(y_hat_list)
        y_hat_finish = self.y_transformer.inverse_transform(y_hat_ensembled)

        return y_hat_finish
    
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader):

        self.model.train()
        
        epoch_statistics = EpochStatistics()

        for batch in dataloader:
            
            x_train, y_train, x_test, y_test = batch
        
            x_train = x_train.to(self.cfg.device)
            y_train = y_train.to(self.cfg.device)
            x_test = x_test.to(self.cfg.device)
            y_test = y_test.to(self.cfg.device)
            
            y_hat = self.model(x_train, y_train, x_test)

            match self.cfg.task:
                case Task.REGRESSION:
                    y_hat = y_hat[:, 0]
                case Task.CLASSIFICATION:
                    y_hat = y_hat[:, :2]

            loss = self.loss(y_hat, y_test)
            score = self.score(y_hat, y_test)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_statistics.update(loss.item(), score, len(y_test))

        loss, score = epoch_statistics.get()
        return loss, score


    def predict_epoch(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:

        self.model.eval()

        y_hat_list = []

        with torch.no_grad():
            for batch in dataloader:
                
                x_train, y_train, x_test, _ = batch

                x_train = x_train.to(self.cfg.device)
                y_train = y_train.to(self.cfg.device)
                x_test = x_test.to(self.cfg.device)
                
                y_hat = self.model(x_train, y_train, x_test)

                match self.cfg.task:
                    case Task.REGRESSION:
                        y_hat = y_hat[:, 0]
                    case Task.CLASSIFICATION:
                        y_hat = y_hat[:, :2]

                y_hat_list.append(y_hat.cpu().numpy())

        y_hat = np.concatenate(y_hat_list)
        return y_hat


    def score(self, y_hat, y):

        with torch.no_grad():
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

        
    def make_dataset_split(self, x_train, y_train):

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

    

    def make_dataset(self, x_train, y_train):

        match self.cfg.task:
            case Task.REGRESSION:
                x_t_train, x_t_valid, y_t_train, y_t_valid = train_test_split(
                    x_train, y_train, test_size=0.2
                )
            case Task.CLASSIFICATION:
                skf = StratifiedKFold(n_splits=5)
                indices = next(skf.split(x_train, y_train))
                x_t_train, x_t_valid = x_train[indices[0]], x_train[indices[1]]
                y_t_train, y_t_valid = y_train[indices[0]], y_train[indices[1]]

        return (
            torch.utils.data.TensorDataset(
                torch.as_tensor(x_t_train),
                torch.as_tensor(y_t_train)
            ),
            torch.utils.data.TensorDataset(
                torch.as_tensor(x_t_valid),
                torch.as_tensor(y_t_valid)
            )
        )
        

    def make_loader(self, dataset, training):

        if isinstance(dataset, torch.utils.data.IterableDataset):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                pin_memory=True,
                collate_fn=lambda x: x[0]
            )
        else:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=training,
                pin_memory=True,
                collate_fn=lambda x: x[0]   # dataloader should not make a batch
            )

    