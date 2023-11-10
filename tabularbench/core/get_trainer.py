import torch
from tabularbench.core.enums import ModelName
from tabularbench.core.trainer import Trainer
from tabularbench.core.trainer_pfn_finetune import Trainer as TrainerPFN

from tabularbench.sweeps.config_run import ConfigRun


def get_trainer(cfg: ConfigRun, model: torch.nn.Module):

    match cfg.model:
        case ModelName.FT_TRANSFORMER:
            return Trainer(cfg, model)
        case ModelName.TABPFN_FINETUNE:
            return TrainerPFN(cfg, model)