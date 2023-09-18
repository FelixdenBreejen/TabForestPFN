from omegaconf import DictConfig
import torch
from torch.optim import AdamW, Adam, SGD


def get_optimizer(hyperparams: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:

    optimizer: torch.optim.Optimizer

    if hyperparams.optimizer == "adam":
        optimizer = Adam(
            model.parameters(), 
            lr=hyperparams.lr,
            betas=(0.9, 0.999),
            weight_decay=hyperparams.weight_decay
        )
    elif hyperparams.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(), 
            lr=hyperparams.lr,
            betas=(0.9, 0.999),
            weight_decay=hyperparams.weight_decay
        )
    elif hyperparams.optimizer == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=hyperparams.lr,
            weight_decay=hyperparams.weight_decay
        )
    else:
        raise ValueError("Optimizer not recognized")
    
    return optimizer