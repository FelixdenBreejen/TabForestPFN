from __future__ import annotations

import os
import sys

import hydra
import torch
import torch.multiprocessing as mp
from loguru import logger
from omegaconf import DictConfig

from main import check_existence_of_benchmark_results_csv
from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.trainer_pretrain import TrainerPretrain
from tabularbench.utils.paths_and_filenames import CONFIG_PRETRAIN_FILE_NAME
from tabularbench.utils.set_seed import set_seed


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(cfg_hydra: DictConfig):
    
    cfg = ConfigPretrain.from_hydra(cfg_hydra)
    barrier = setup_multiprocessing(cfg)
    setup_logger(cfg)

    check_existence_of_benchmark_results_csv(cfg)
    cfg.save(path=cfg.output_dir / CONFIG_PRETRAIN_FILE_NAME)
    setup_gpus(cfg)
    set_seed(cfg.seed)

    logger.info(f"Training with {len(cfg.devices)} GPU(s)")
    mp.spawn(main_experiment, nprocs=len(cfg.devices), args=(cfg,barrier))


def main_experiment(gpu: int, cfg: ConfigPretrain, barrier: mp.Barrier) -> None:

    logger.add(cfg.output_dir / "log.log", enqueue=True)

    setup_gpus_of_experiment(cfg, gpu)
    
    trainer = TrainerPretrain(cfg, barrier)

    if cfg.is_main_process:
        logger.info(f"Trainer of {cfg.model_name.value} created, start training")
    
    trainer.train()

    if cfg.is_main_process:
        logger.info(f"Finished training of {cfg.model_name.value}")
        logger.info(f"Start testing of {cfg.model_name.value}")
        trainer.test()
        logger.info(f"Finished testing of {cfg.model_name.value}")


def setup_multiprocessing(cfg: ConfigPretrain) -> mp.Barrier:

    mp.set_start_method('spawn')

    if debugger_is_active():
        os.environ['CUDA_LAUNCH_BLOCKING']='1'

    return mp.Barrier(len(cfg.devices))


def setup_logger(cfg: ConfigPretrain) -> None:
    # Should be called after setting up the multiprocessing, because enqueue is used

    logger.add(cfg.output_dir / "log.log", enqueue=True)
    logger.info("Finished creating pretrain config")


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
        port = 5678 + cfg.devices[0]
        os.environ['MASTER_PORT'] = str(port)
        
        torch.distributed.init_process_group(backend="nccl", world_size = len(cfg.devices), rank=gpu)

    return device


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


if __name__ == "__main__":
    main()