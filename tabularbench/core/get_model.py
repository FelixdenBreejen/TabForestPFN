
import torch
import numpy as np

from tabularbench.core.enums import ModelName
from tabularbench.sweeps.run_config import RunConfig
from tabularbench.models.ft_transformer.ft_transformer import FTTransformer


def get_model(cfg: RunConfig, x_train: np.ndarray, y_train: np.ndarray, categorical_indicator: np.ndarray) -> torch.nn.Module:

    match cfg.model:
        case ModelName.FT_TRANSFORMER:
            return FTTransformer(cfg, x_train, y_train, categorical_indicator)
        case _:
            raise NotImplementedError(f"Model {cfg.model} not implemented yet")
