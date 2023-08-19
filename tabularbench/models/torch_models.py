import torch.nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
import sys
sys.path.append("")
from tabularbench.models.tabular.bin.resnet import ResNet, InputShapeSetterResnet
from tabularbench.models.tabular.bin.mlp import MLP, InputShapeSetterMLP
from tabularbench.models.tabular.bin.mlp_pwl import MLP_PWL, InputShapeSetterMLP_PWL
from tabularbench.models.tabular.bin.ft_transformer import Transformer, InputShapeSetterTransformer

from tabularbench.core.trainer import Trainer

from skorch.callbacks import Callback
import numpy as np


def modify_config(model_config):

    if "lr_scheduler" not in model_config:
        model_config['lr_scheduler'] = False
        
    if "es_patience" not in model_config:
        model_config['es_patience'] = 40
        
    if "lr_patience" not in model_config:
        model_config['lr_patience'] = 30
        
    if "categories" not in model_config:
        model_config['categories'] = None


def create_ft_transformer_torch(model_config, id, use_checkpoints=True):

    model_config = {**model_config}
    
    modify_config(model_config)

    trainer = Trainer(
        model=Transformer,
        input_shape_setter=InputShapeSetterTransformer,
        model_config=model_config,
    )

    return trainer

    model_skorch = NeuralNetClassifier(
        Transformer,
        # Shuffle training data on each epoch
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(batch_size, 1),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_numerical=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=categorical_indicator,
        module__feature_representation_list=None,
        verbose=0,
        callbacks=callbacks,
        **kwargs
    )

    return model_skorch
