import torch
from omegaconf import DictConfig
from torch.optim import SGD, Adam, AdamW

from tabularbench.config.config_pretrain import ConfigPretrain


def get_optimizer(hyperparams: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:

    optimizer: torch.optim.Optimizer

    if hyperparams['optimizer'] == "adam":
        optimizer = Adam(
            model.parameters(), 
            lr=hyperparams['lr'],
            betas=(0.9, 0.999),
            weight_decay=hyperparams['weight_decay']
        )
    elif hyperparams['optimizer'] == "adamw":
        optimizer = AdamW(
            model.parameters(), 
            lr=hyperparams['lr'],
            betas=(0.9, 0.999),
            weight_decay=hyperparams['weight_decay']
        )
    elif hyperparams['optimizer'] == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=hyperparams['lr'],
            weight_decay=hyperparams['weight_decay']
        )
    else:
        raise ValueError("Optimizer not recognized")
    
    return optimizer


def get_optimizer_pretrain(cfg: ConfigPretrain, model: torch.nn.Module) -> torch.optim.Optimizer:

    parameters = [(name, param) for name, param in model.named_parameters()]

    parameters_with_weight_decay = []
    parameters_without_weight_decay = []

    for name, param in parameters:
        if name.endswith("bias") or 'norm' in name:
            parameters_without_weight_decay.append(param)
        else:
            parameters_with_weight_decay.append(param)

    optimizer_parameters = [
        {"params": parameters_with_weight_decay, "weight_decay": cfg.optim.weight_decay},
        {"params": parameters_without_weight_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.Adam(
        optimizer_parameters, 
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        weight_decay=cfg.optim.weight_decay
    )
    
    return optimizer