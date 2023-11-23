from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch
from tabularbench.core.enums import ModelName


from tabularbench.sweeps.config_benchmark_sweep import ConfigPlotting
from tabularbench.sweeps.get_logger import get_logger


@dataclass
class ConfigPretrain():
    logger: logging.Logger
    output_dir: Path
    seed: int
    devices: list[torch.device]
    model: DictConfig
    data: DictConfig
    optim: DictConfig
    plotting: ConfigPlotting


    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        output_dir = Path(cfg_hydra.output_dir)
        logger = get_logger(output_dir / 'log.txt')

        devices = [torch.device(device) for device in cfg_hydra.devices]

        return cls(
            logger=logger,
            output_dir=output_dir,
            devices=devices,
            seed=cfg_hydra.seed,
            model = cfg_hydra.model,
            data = cfg_hydra.data,
            optim = cfg_hydra.optim,
            plotting = ConfigPlotting(
                n_runs=cfg_hydra.plotting.n_runs,
                n_random_shuffles=cfg_hydra.plotting.n_random_shuffles,
                confidence_bound=cfg_hydra.plotting.confidence_bound,
                plot_default_value=cfg_hydra.plotting.plot_default_value,
                benchmark_model_names=[ModelName[model] for model in cfg_hydra.plotting.benchmark_models],
            ),
        )
    






