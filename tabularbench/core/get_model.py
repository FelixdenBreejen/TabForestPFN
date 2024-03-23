
import numpy as np
import torch

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.config.config_run import ConfigRun
from tabularbench.core.enums import ModelName
from tabularbench.models.foundation.foundation_transformer import FoundationTransformer
from tabularbench.models.ft_transformer.ft_transformer import FTTransformer
from tabularbench.models.tabPFN.tabpfn_transformer import TabPFN


def get_model(cfg: ConfigRun, x_train: np.ndarray, y_train: np.ndarray, categorical_indicator: np.ndarray) -> torch.nn.Module:

    match cfg.model_name:
        case ModelName.FT_TRANSFORMER:
            return FTTransformer(cfg, x_train, y_train, categorical_indicator)
        case ModelName.TABPFN:
            return TabPFN(
                cfg.hyperparams['use_pretrained_weights'], 
                path_to_weights=cfg.hyperparams['path_to_weights']
            )
        case ModelName.FOUNDATION:
            return FoundationTransformer(
                n_features=cfg.hyperparams['n_features'],
                n_classes=cfg.hyperparams['n_classes'],
                dim=cfg.hyperparams['dim'],
                n_layers=cfg.hyperparams['n_layers'],
                n_heads=cfg.hyperparams['n_heads'],
                attn_dropout=cfg.hyperparams['attn_dropout'],
                y_as_float_embedding=cfg.hyperparams['y_as_float_embedding'],
                use_pretrained_weights=cfg.hyperparams['use_pretrained_weights'],
                path_to_weights=cfg.hyperparams['path_to_weights']
            )
        case _:
            raise NotImplementedError(f"Model {cfg.model_name} not implemented yet")
            
        


def get_model_pretrain(cfg: ConfigPretrain) -> torch.nn.Module:

    match cfg.model_name:
        case ModelName.TABPFN:
            return TabPFN(
                use_pretrained_weights=cfg.optim.use_pretrained_weights,
                path_to_weights=cfg.optim.path_to_weights
            )
        case ModelName.FOUNDATION:
            return FoundationTransformer(
                n_features=cfg.data.max_features,
                n_classes=cfg.data.max_classes,
                dim=cfg.model['dim'],
                n_layers=cfg.model['n_layers'],
                n_heads=cfg.model['n_heads'],
                attn_dropout=cfg.model['attn_dropout'],
                y_as_float_embedding=cfg.model['y_as_float_embedding'],
                use_pretrained_weights=cfg.optim.use_pretrained_weights,
                path_to_weights=cfg.optim.path_to_weights
            )
        case _:
            raise NotImplementedError(f"Model {cfg.model['name']} not implemented yet")
