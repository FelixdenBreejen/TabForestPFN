from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np

from tabularbench.core.callbacks import EarlyStopping, Checkpoint, EpochStatistics


class Trainer(BaseEstimator):

    def __init__(
            self, 
            Model: type[torch.nn.Module], 
            InputShapeSetter: type[torch.nn.Module], 
            model_config: dict
        ) -> None:

        self.InputShapeSetterClass = InputShapeSetter
        self.ModelClass = Model
        self.cfg = model_config

        self.early_stopping = EarlyStopping(patience=self.cfg['es_patience'])
        self.checkpoint = Checkpoint("temp_weights", self.cfg['id'])
        
        if self.cfg['categorical_indicator'] is not None:
            self.categorical_indicator = torch.BoolTensor(self.cfg['categorical_indicator'])



    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        input_shape_setter = self.InputShapeSetterClass(
            categorical_indicator = self.cfg['categorical_indicator'],
            regression = self.cfg['regression'],
            batch_size = self.cfg['batch_size'],
            categories = self.cfg['categories'],
        )

        input_shape_config = input_shape_setter.on_train_begin(None, x_train, y_train)

        module_config = extract_module_config(self.cfg, input_shape_config)

        self.model = self.ModelClass(**module_config).cuda()
        self.loss = self.select_loss().cuda()

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        dataset_train, dataset_valid = self.make_dataset(x_train=x_train, y_train=y_train)
        loader_train = self.make_loader(dataset_train, training=True)
        loader_valid = self.make_loader(dataset_valid, training=False)

        self.train(loader_train, loader_valid)

        return self


    def train(self, loader_train, loader_valid):

        for epoch in range(self.cfg['max_epochs']):
            
            self.model.train()
        
            epoch_statistics_train = EpochStatistics()

            for batch in loader_train:

                x, y = batch
                x = x.cuda()
                y = y.cuda()
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
                    x = x.cuda()
                    y = y.cuda()
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
                x = batch[0].cuda()
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
    

    def make_dataset(self, x_train, y_train):

        if self.cfg['regression']:
            x_t_train, x_t_valid, y_t_train, y_t_valid = train_test_split(
                x_train, y_train, test_size=0.2
            )
        else:
            skf = StratifiedKFold(n_splits=5)
            indices = next(skf.split(x_train, y_train))
            x_t_train, x_t_valid = x_train[indices[0]], x_train[indices[1]]
            y_t_train, y_t_valid = y_train[indices[0]], y_train[indices[1]]


        if self.cfg['regression']:
            return (
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_t_train),
                    torch.FloatTensor(y_t_train)
                ), 
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_t_valid),
                    torch.FloatTensor(y_t_valid)
                )
            )
        else:
            return (
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_t_train),
                    torch.LongTensor(y_t_train)
                ), 
                torch.utils.data.TensorDataset(
                    torch.FloatTensor(x_t_valid),
                    torch.LongTensor(y_t_valid)
                )
            )
        

    def make_loader(self, dataset, training):

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=training,
            pin_memory=True,
        )

    
    def select_loss(self):

        if self.cfg['regression']:
            loss = torch.nn.MSELoss()
        else:
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