from omegaconf import DictConfig
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR



def get_scheduler(hyperparams: DictConfig, optimizer: torch.optim.Optimizer):

    if hyperparams.lr_scheduler:      
        scheduler = ReduceLROnPlateau(
            optimizer, 
            patience=hyperparams.lr_scheduler_patience, 
            min_lr=2e-5, 
            factor=0.2
        )
    else:
        scheduler = LambdaLR(       # type: ignore
            optimizer,
            lambda _: 1
        )

    return scheduler