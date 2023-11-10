
import torch
import numpy as np

from tabularbench.core.enums import ModelName
from tabularbench.models.tabPFN.transformer import TabPFN
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.models.ft_transformer.ft_transformer import FTTransformer


def get_model(cfg: ConfigRun, x_train: np.ndarray, y_train: np.ndarray, categorical_indicator: np.ndarray) -> torch.nn.Module:

    match cfg.model_name:
        case ModelName.FT_TRANSFORMER:
            return FTTransformer(cfg, x_train, y_train, categorical_indicator)
        case ModelName.TABPFN_FINETUNE:
            return TabPFN(cfg)
        case _:
            raise NotImplementedError(f"Model {cfg.model_name} not implemented yet")
