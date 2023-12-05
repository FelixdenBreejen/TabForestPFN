import torch
from tabularbench.core.enums import ModelName
from tabularbench.core.trainer import Trainer
from tabularbench.core.trainer_finetune import TrainerFinetune

from tabularbench.sweeps.config_run import ConfigRun


def get_trainer(cfg: ConfigRun, model: torch.nn.Module):

    match cfg.model_name:
        case ModelName.FT_TRANSFORMER:
            return Trainer(cfg, model)
        case ModelName.TABPFN | ModelName.FOUNDATION:
            return TrainerFinetune(cfg, model)