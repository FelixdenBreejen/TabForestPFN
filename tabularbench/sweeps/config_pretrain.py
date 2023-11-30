from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

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
    use_ddp: bool
    workers_per_gpu: int
    model: DictConfig
    data: ConfigData
    optim: ConfigOptim
    plotting: ConfigPlotting
    hyperparams_finetuning: DictConfig

    device: Optional[torch.device] = None   # initialized later by ddp
    is_main_process: bool = True            # initialized later by ddp


    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        output_dir = Path(cfg_hydra.output_dir)
        logger = get_logger(output_dir / 'log.txt')

        devices = [torch.device(device) for device in cfg_hydra.devices]

        return cls(
            logger=logger,
            output_dir=output_dir,
            devices=devices,
            use_ddp=len(devices) > 1,
            seed=cfg_hydra.seed,
            workers_per_gpu=cfg_hydra.workers_per_gpu,
            model = cfg_hydra.model,
            data = ConfigData(
                min_samples_support=cfg_hydra.data.min_samples_support,
                max_samples_support=cfg_hydra.data.max_samples_support,
                n_samples_query=cfg_hydra.data.n_samples_query,
                min_features=cfg_hydra.data.min_features,
                max_features=cfg_hydra.data.max_features,
                max_classes=cfg_hydra.data.max_classes,
            ),
            optim = ConfigOptim(
                max_steps=cfg_hydra.optim.max_steps,
                log_every_n_steps=cfg_hydra.optim.log_every_n_steps,
                eval_every_n_steps=cfg_hydra.optim.eval_every_n_steps,
                batch_size=cfg_hydra.optim.batch_size,
                gradient_accumulation_steps=cfg_hydra.optim.gradient_accumulation_steps,
                lr=cfg_hydra.optim.lr,
                weight_decay=cfg_hydra.optim.weight_decay,
                beta1=cfg_hydra.optim.beta1,
                beta2=cfg_hydra.optim.beta2,
                warmup_steps=cfg_hydra.optim.warmup_steps,
                cosine_scheduler=cfg_hydra.optim.cosine_scheduler,
            ),
            plotting = ConfigPlotting(
                n_runs=cfg_hydra.plotting.n_runs,
                n_random_shuffles=cfg_hydra.plotting.n_random_shuffles,
                confidence_bound=cfg_hydra.plotting.confidence_bound,
                plot_default_value=cfg_hydra.plotting.plot_default_value,
                benchmark_model_names=[ModelName[model] for model in cfg_hydra.plotting.benchmark_models],
            ),
            hyperparams_finetuning = cfg_hydra.hyperparams.tabpfn_finetune
        )
    

@dataclass
class ConfigOptim():
    max_steps: int
    log_every_n_steps: int
    eval_every_n_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    cosine_scheduler: bool
    max_grad_norm: float


@dataclass
class ConfigData():
    min_samples_support: int
    max_samples_support: int
    n_samples_query: int
    min_features: int
    max_features: int
    max_classes: int







