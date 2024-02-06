from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from main import check_existence_of_benchmark_results_csv
from tabularbench.core.trainer_pretrain import TrainerPretrain
from tabularbench.utils.config_pretrain import ConfigPretrain
from tabularbench.utils.paths_and_filenames import CONFIG_DUPLICATE
from tabularbench.utils.set_seed import set_seed


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(cfg_hydra: DictConfig):
    
    mp.set_start_method('spawn')

    if debugger_is_active():
        os.environ['CUDA_LAUNCH_BLOCKING']='1'

    cfg = ConfigPretrain.from_hydra(cfg_hydra)
    logger.add(cfg.output_dir / "log.log", enqueue=True)
    logger.info("Finished creating pretrain config")

    check_existence_of_benchmark_results_csv(cfg)
    cfg.save()
    setup_gpus(cfg)
    set_seed(cfg.seed)

    barrier = mp.Barrier(len(cfg.devices))

    if cfg.use_ddp:
        logger.info(f"Training with {len(cfg.devices)} GPUs")
        mp.spawn(main_experiment, nprocs=len(cfg.devices), args=(cfg,barrier))
    else:
        logger.info(f"Training with one GPU")
        mp.spawn(main_experiment, nprocs=1, args=(cfg,barrier))


def main_experiment(gpu: int, cfg: ConfigPretrain, barrier: mp.Barrier) -> None:

    setup_gpus_of_experiment(cfg, gpu)
    
    trainer = TrainerPretrain(cfg, barrier)

    if cfg.is_main_process:
        logger.info(f"Trainer of {cfg.model.name.value} created, start training")
    
    trainer.train()

    if cfg.is_main_process:
        logger.info(f"Finished training of {cfg.model.name.value}")
        logger.info(f"Start testing of {cfg.model.name.value}")
        trainer.test()
        logger.info(f"Finished testing of {cfg.model.name.value}")


def setup_gpus(cfg: ConfigPretrain) -> None:

    num_gpus = len(cfg.devices)    

    if cfg.use_ddp:

        if num_gpus == 1:
            logger.info("Are you sure you want distributed training with only one GPU?")

        batch_size = cfg.optim.batch_size
        assert batch_size >= num_gpus, "Batch size must be at least the number of GPUs"
        cfg.optim.batch_size = batch_size // num_gpus if cfg.use_ddp else batch_size

        logger.info(f"Using GPUs {[d.index for d in cfg.devices]} for distributed training")
        logger.info(f"Batch size per device set to {cfg.optim.batch_size}")
        logger.info(f"With gradient accumulation steps {cfg.optim.gradient_accumulation_steps}, total batch size is {cfg.optim.batch_size * cfg.optim.gradient_accumulation_steps * num_gpus}")

    else:
        assert num_gpus == 1, "Cannot use more than one GPU without distributed training"
        cfg.device = cfg.devices[0]
        cfg.is_main_process = True
        logger.info(f"Using GPU {cfg.device} for training")


def setup_gpus_of_experiment(cfg: ConfigPretrain, gpu: int) -> torch.device:

    device = cfg.devices[gpu]
    torch.cuda.set_device(device)
    cfg.device = device
    cfg.is_main_process = (gpu == 0)

    if cfg.use_ddp:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['MASTER_ADDR'] = 'localhost'
        port = 5678 + cfg.devices[0].index
        os.environ['MASTER_PORT'] = str(port)
        
        torch.distributed.init_process_group(backend="nccl", world_size = len(cfg.devices), rank=gpu)

    return device


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None




def save_config(cfg: ConfigPretrain) -> None:
    
    config_path = Path(cfg.output_dir) / CONFIG_DUPLICATE

    # OmegaConf object looks ugly when saved as yaml
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    finetuning_dict = OmegaConf.to_container(cfg.hyperparams_finetuning, resolve=True)
    self_to_save = dataclasses.replace(cfg, model=model_dict, hyperparams_finetuning=finetuning_dict)

    with open(config_path, 'w') as f:        
        yaml.dump(self_to_save, f, default_flow_style=False)


if __name__ == "__main__":
    main()