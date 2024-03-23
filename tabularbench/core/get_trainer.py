import torch

from tabularbench.config.config_run import ConfigRun
from tabularbench.core.enums import ModelName
from tabularbench.core.trainer import Trainer
from tabularbench.core.trainer_finetune import TrainerFinetune


def get_trainer(cfg: ConfigRun, model: torch.nn.Module, n_classes: int):

    match cfg.model_name:
        case ModelName.FT_TRANSFORMER:
            return Trainer(cfg, model, n_classes)
        case ModelName.TABPFN | ModelName.FOUNDATION:
            return TrainerFinetune(cfg, model, n_classes)