from __future__ import annotations
from pathlib import Path
import sys
from omegaconf import OmegaConf, DictConfig
import logging

import random
import numpy as np
import torch

from tabularbench.sweeps.paths_and_filenames import CONFIG_DUPLICATE


def get_config(output_dir: str) -> DictConfig:
    return OmegaConf.load(Path(output_dir) / CONFIG_DUPLICATE)
    

def get_logger(cfg: OmegaConf, log_file_name) -> logging.Logger:

    logging.setLogRecordFactory(CustomLogRecord)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)-8s :: %(funcNameMaxWidth)-18s ::   %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    log_dir = Path(cfg['output_dir']) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / log_file_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.funcNameMaxWidth = self.funcName[:15] + '...' if len(self.funcName) > 18 else self.funcName

    
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_device_to_cfg(cfg: dict, gpu: int) -> None:
    cfg.device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'