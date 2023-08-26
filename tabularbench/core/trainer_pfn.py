from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np

from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics
from tabularbench.models.tabPFN.load_model import load_pretrained_model
from tabularbench.models.tabPFN.dataset import TabPFNDataset

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
            self.categorical_indicator = torch.BoolTensor(self.cfg['categorical_indicator'])



    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        self.model, pretrain_config = load_pretrained_model()
        self.model.to(self.cfg['device'])

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        self.x_train = x_train
        self.y_train = y_train

        a = self.make_dataset_split(x_train=x_train, y_train=y_train)
        self.x_train_train, self.x_train_valid, self.y_train_train, self.y_train_valid = a


        # self.train(loader_train, loader_valid)

        return self


    def train(self, loader_train, loader_valid):

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

        y_hat_list = []

        dataset = TabPFNDataset(self.x_train, self.y_train, x, batch_size=self.cfg['batch_size'])
        loader = self.make_loader(dataset, training=False)
        
        with torch.no_grad():
            for _ in range(self.cfg['n_ensembles']):
                y_hat_pieces = []

                for input in loader:

                    # classif = TabPFNClassifier(device=self.cfg['device'], N_ensemble_configurations=3)
                    # classif.fit(input[0], input[1])
                    # output = classif.predict_proba(input[2])
                    
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


    def select_optimizer(self):

        optimizer_name = self.cfg['optimizer']

        if optimizer_name == "adam":
            optimizer = Adam(
                self.model.parameters(), 
                lr=self.cfg['lr'],
                betas=(0.9, 0.999),
                weight_decay=self.cfg['optimizer__weight_decay']
            )
        elif optimizer_name == "adamw":
            optimizer = AdamW(
                self.model.parameters(), 
                lr=self.cfg['lr'],
                betas=(0.9, 0.999),
                weight_decay=self.cfg['optimizer__weight_decay']
            )
        elif optimizer_name == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg['lr'],
                weight_decay=self.cfg['optimizer__weight_decay']
            )
        else:
            raise ValueError("Optimizer not recognized")
        
        return optimizer
        

    def select_scheduler(self):

        if self.cfg['lr_scheduler']:                
            scheduler = ReduceLROnPlateau(
                self.optimizer, 
                patience=self.cfg['lr_patience'], 
                min_lr=2e-5, 
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