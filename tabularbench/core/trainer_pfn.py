from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics
from tabularbench.models.tabPFN.load_model import load_pretrained_model
from tabularbench.models.tabPFN.dataset import TabPFNDataset, TabPFNDatasetGenerator

from tabpfn import TabPFNClassifier


class TrainerPFN(BaseEstimator):

    def __init__(
            self, 
            model_config: dict
        ) -> None:

        
        self.cfg = model_config

        self.early_stopping = EarlyStopping(patience=self.cfg['es_patience'])
        self.checkpoint = Checkpoint("temp_weights", self.cfg['id'])
        
        if self.cfg['categorical_indicator'] is not None:
            self.categorical_indicator = self.cfg['categorical_indicator']

        self.onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')



    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        self.model, pretrain_config = load_pretrained_model()
        self.model.to(self.cfg['device'])

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        self.x_train = x_train
        self.y_train = y_train

        self.onehot_encoder.fit(self.x_train[:, self.categorical_indicator])
        # self.x_train = self.onehot_encode(self.x_train, max_features=100)

        a = self.make_dataset_split(x_train=x_train, y_train=y_train)
        self.x_train_train, self.x_train_valid, self.y_train_train, self.y_train_valid = a

        dataset_train_generator = TabPFNDatasetGenerator(
            self.x_train_train,
            self.y_train_train,
            batch_size=self.cfg['batch_size'],
        )

        dataset_valid = TabPFNDataset(
            self.x_train_train, 
            self.y_train_train, 
            self.x_train_valid, 
            self.y_train_valid,
            batch_size=self.cfg['batch_size']
        )

        loader_valid = self.make_loader(dataset_valid, training=False)

        self.loss = self.select_loss()
        self.optimizer = self.select_optimizer()

        self.train(dataset_train_generator, dataset_valid, loader_valid)

        return self


    def train(self, dataset_train_generator, dataset_valid, loader_valid):

        for epoch in range(self.cfg['max_epochs']):

            dataset_train = next(dataset_train_generator)            
            loader_train = self.make_loader(dataset_train, training=True)
            
            self.model.train()
        
            epoch_statistics_train = EpochStatistics()

            for batch in loader_train:

                input, y_test = batch
                input = tuple(x.to(self.cfg['device']) for x in input)
                y_test = y_test.to(self.cfg['device'])

                y_hat_train = self.model(input, single_eval_pos=dataset_train.single_eval_pos)
                loss = self.loss(y_hat_train, y_test)
                score = self.score(y_hat_train, y_test)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_statistics_train.update(loss.item(), score, len(y_test))

            loss_train, score_train = epoch_statistics_train.get()

            self.model.eval()

            epoch_statistics_valid = EpochStatistics()
            
            with torch.no_grad():

                for batch in loader_valid:

                    input, y_test = batch
                    input = tuple(x.to(self.cfg['device']) for x in input)
                    y_test = y_test.to(self.cfg['device'])
                    
                    y_hat_valid = self.model(input, single_eval_pos=dataset_valid.single_eval_pos)
                    loss_valid = self.loss(y_hat_valid, y_test)
                    score_valid = self.score(y_hat_valid, y_test)
                    
                    epoch_statistics_valid.update(loss_valid.item(), score_valid, len(y_test))

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

        # x = self.onehot_encode(x, max_features=100)

        y_hat_list = []

        dataset = TabPFNDataset(self.x_train, self.y_train, x, batch_size=self.cfg['batch_size'])
        loader = self.make_loader(dataset, training=False)

        with torch.no_grad():
            for _ in range(self.cfg['n_ensembles']):
                y_hat_pieces = []

                for input in loader:
                    
                    input = tuple(x.to(self.cfg['device']) for x in input)
                    output = self.model(input, single_eval_pos=dataset.single_eval_pos)
                    output = output.cpu().numpy()

                    y_hat_pieces.append(output)

                y_hat_list.append(np.concatenate(y_hat_pieces))

        y_hat = sum(y_hat_list) / len(y_hat_list)

        if self.cfg['regression']:
            return y_hat
        else:
            return y_hat.argmax(axis=1)


    def score(self, y_hat, y):

        with torch.no_grad():
            if self.cfg['regression']:  
                return np.sqrt(np.mean((y_hat.cpu().numpy() - y.cpu().numpy())**2))
            else:
                return np.mean((y_hat.cpu().numpy().argmax(axis=1) == y.cpu().numpy()))
            

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))

    
    def onehot_encode(self, x: np.ndarray, max_features: int):

        x_categorical = self.onehot_encoder.transform(x[:, self.categorical_indicator])
        x_numerical = x[:, ~self.categorical_indicator]

        x_new = np.concatenate([x_numerical, x_categorical], axis=1)

        if x_new.shape[1] < max_features:
            return x_new
        else:
            print("Warning: onehot-encoded features exceed max_features. Returning original features.")
            return x


    def select_optimizer(self):

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.cfg['lr'],
            betas=(0.9, 0.999),
            weight_decay=self.cfg['optimizer__weight_decay']
        )
        
        return optimizer
        

    def select_scheduler(self):

        if self.cfg['lr_scheduler']:                
            scheduler = ReduceLROnPlateau(
                self.optimizer, 
                patience=self.cfg['lr_patience'],
                factor=0.2
            )
        else:
            scheduler = LambdaLR(
                self.optimizer,
                lambda _: 1
            )

        return scheduler
    

    def make_dataset_split(self, x_train, y_train):

        if self.cfg['regression']:
            x_t_train, x_t_valid, y_t_train, y_t_valid = train_test_split(
                x_train, y_train, test_size=0.2
            )
        else:
            skf = StratifiedKFold(n_splits=5)
            indices = next(skf.split(x_train, y_train))
            x_t_train, x_t_valid = x_train[indices[0]], x_train[indices[1]]
            y_t_train, y_t_valid = y_train[indices[0]], y_train[indices[1]]

        return x_t_train, x_t_valid, y_t_train, y_t_valid
        

    def make_loader(self, dataset, training):

        if isinstance(dataset, torch.utils.data.IterableDataset):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                pin_memory=True,
                collate_fn=lambda x: x[0]
            )
        else:
            # dataloader should not make a batch
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=training,
                pin_memory=True,
                collate_fn=lambda x: x[0]
            )

    
    def select_loss(self):

        if self.cfg['regression']:
            loss = torch.nn.MSELoss()
        else:
            loss = torch.nn.CrossEntropyLoss()

        return loss