
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import numpy as np



class Trainer(BaseEstimator):

    def __init__(
            self, 
            Model: type[torch.Module], 
            InputShapeSetter: type[torch.Module], 
            model_config: dict
        ) -> None:

        self.InputShapeSetterClass = InputShapeSetter
        self.ModelClass = Model
        self.cfg = model_config

        
        # EarlyStopping(monitor="valid_loss", patience=es_patience)
        # EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True)
        # EpochScoring(scoring='accuracy', name='train_accuracy', on_train=True) # FIXME make customizable
        # Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None, f_criterion=None))

        
        if not categorical_indicator is None:
            categorical_indicator = torch.BoolTensor(categorical_indicator)


        pass

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):

        input_shape_setter = self.InputShapeSetterClass(
            categorical_indicator = self.cfg['categorical_indicator'],
            regression = self.cfg['regression'],
            batch_size = self.cfg['batch_size'],
            categories = self.cfg['categories'],
        )

        input_shape_config = input_shape_setter.on_train_begin(None, x_train, y_train)

        module_config = extract_module_config(self.cfg, input_shape_config)

        self.model = self.ModelClass(**module_config)

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        self.train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_train),
            torch.FloatTensor(y_train)
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
        )

        self.train()

        return self


    def train(self):

        for batch in self.train_loader:

            x, y = batch
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()


        

    def predict(x):
        pass


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
                lambda: 1
            )

        return scheduler


def extract_module_config(model_config, input_shape_config):

    module_config = {}
    total_config = {**model_config, **input_shape_config}

    for key in total_config.keys():
        if key.startswith("module__"):
            module_config[key[len("module__"):]] = model_config[key]

    return module_config