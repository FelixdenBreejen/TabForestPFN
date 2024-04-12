import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from tabularbench.config.config_pretrain import ConfigPretrain


def get_scheduler(hyperparams: dict, optimizer: torch.optim.Optimizer):

    if hyperparams['lr_scheduler']:      
        scheduler = ReduceLROnPlateau(
            optimizer, 
            patience=hyperparams['lr_scheduler_patience'], 
            min_lr=0, 
            factor=0.2
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, 
            patience=10000000, 
            min_lr=0, 
            factor=0.2
        )

    return scheduler


def get_scheduler_pretrain(cfg: ConfigPretrain, optimizer: torch.optim.Optimizer):

    
    if cfg.optim.cosine_scheduler:
        schedule = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.warmup_steps,
            num_training_steps=cfg.optim.max_steps
        )
    else:
        schedule = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.optim.warmup_steps
        )

    return schedule