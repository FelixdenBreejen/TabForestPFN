
import torch
import numpy as np

from tabularbench.sweeps.run_config import RunConfig

from tabularbench.models.ft_transformer.ft_transformer import FTTransformer


module_dict = {
    "ft_transformer": FTTransformer,
}


def get_model(cfg: RunConfig, x_train: np.ndarray, y_train: np.ndarray, categorical_indicator: np.ndarray) -> torch.nn.Module:

    check_correct_model_name(cfg.model)
    model = module_dict[cfg.model](cfg, x_train, y_train, categorical_indicator)
    return model


def check_correct_model_name(name: str) -> None:
    
    assert name in module_dict.keys(), f"Model {name} not found in {module_dict.keys()}"