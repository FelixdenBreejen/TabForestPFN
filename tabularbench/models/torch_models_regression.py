import torch.nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
import sys
sys.path.append("")
from tabularbench.models.tabular.bin.resnet import ResNet, InputShapeSetterResnet
from tabularbench.models.tabular.bin.mlp import MLP, InputShapeSetterMLP
from tabularbench.models.tabular.bin.mlp_pwl import MLP_PWL, InputShapeSetterMLP_PWL
from tabularbench.models.tabular.bin.ft_transformer import Transformer, InputShapeSetterTransformer
from tabularbench.models.skorch_models import LearningRateLogger


class NeuralNetRegressorBis(NeuralNetRegressor):
    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return super().fit(X, y)




def create_ft_transformer_regressor_torch(id, wandb_run=None, use_checkpoints=True,
                                           categorical_indicator=None, **kwargs):
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop('es_patience')
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop('lr_patience')
    optimizer = kwargs.pop('optimizer')
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    batch_size = kwargs.pop('batch_size')
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop('categories')
    callbacks = [InputShapeSetterTransformer(regression=True,
                                        categorical_indicator=categorical_indicator,
                                        categories=categories),
                       EarlyStopping(monitor="valid_loss", patience=es_patience)] #TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring='neg_root_mean_squared_error', name='train_accuracy', on_train=True))
    if lr_scheduler:
        callbacks.append(LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)) #FIXME make customizable
    if use_checkpoints:
        callbacks.append(Checkpoint(dirname="skorch_cp", f_params=r"params_{}.pt".format(id), f_optimizer=None,
                                  f_criterion=None))
    if not wandb_run is None:
        callbacks.append(LearningRateLogger())

    if not categorical_indicator is None:
        categorical_indicator = torch.BoolTensor(categorical_indicator)


    model_skorch = NeuralNetRegressorBis(
        Transformer,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(batch_size, 1), # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None, # will be change when fitted
        module__d_out=1,  # idem
        verbose=0,
        callbacks=callbacks,
        module__regression=True,
        module__categorical_indicator=categorical_indicator,
        **kwargs
    )

    return model_skorch
