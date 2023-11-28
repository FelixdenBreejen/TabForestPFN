from __future__ import annotations
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig
import torch
import torch.multiprocessing as mp
from main import check_existence_of_benchmark_results_csv, save_config
from tabularbench.core.trainer_pfn import TrainerPFN
from tabularbench.sweeps.config_pretrain import ConfigPretrain
from tabularbench.sweeps.get_logger import get_logger

from tabularbench.sweeps.set_seed import set_seed



@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(cfg_hydra: DictConfig):
    
    mp.set_start_method('spawn')

    cfg = ConfigPretrain.from_hydra(cfg_hydra)
    cfg.logger.info("Finished creating pretrain config")

    check_existence_of_benchmark_results_csv(cfg)
    save_config(cfg)
    setup_gpus(cfg)
    set_seed(cfg.seed)

    barrier = mp.Barrier(len(cfg.devices))

    if cfg.use_ddp:
        mp.spawn(main_experiment, nprocs=len(cfg.devices), args=(cfg,barrier))
    else:
        mp.spawn(main_experiment, nprocs=1, args=(cfg,barrier))


def main_experiment(gpu: int, cfg: ConfigPretrain, barrier: mp.Barrier) -> None:

    # reset logger 
    cfg.logger = get_logger(Path(cfg.logger.name))

    setup_gpus_of_experiment(cfg, gpu)
    
    trainer = TrainerPFN(cfg, barrier)
    trainer.train()

    if cfg.is_main_process:
        trainer.test()


def setup_gpus(cfg: ConfigPretrain) -> None:

    num_gpus = len(cfg.devices)    

    if cfg.use_ddp:

        if num_gpus == 1:
            cfg.logger.info("Are you sure you want distributed training with only one GPU?")

        batch_size = cfg.optim.batch_size
        assert batch_size >= num_gpus, "Batch size must be at least the number of GPUs"
        cfg.optim.batch_size = batch_size // num_gpus if cfg.use_ddp else batch_size

        cfg.logger.info(f"Using GPUs {[d.index for d in cfg.devices]} for distributed training")
        cfg.logger.info(f"Batch size per device set to {cfg.optim.batch_size}")
        cfg.logger.info(f"With gradient accumulation steps {cfg.optim.gradient_accumulation_steps}, total batch size is {cfg.optim.batch_size * cfg.optim.gradient_accumulation_steps * num_gpus}")

    else:
        assert num_gpus == 1, "Cannot use more than one GPU without distributed training"
        cfg.device = cfg.devices[0]
        cfg.is_main_process = True
        cfg.logger.info(f"Using GPU {cfg.device} for training")


def setup_gpus_of_experiment(cfg: ConfigPretrain, gpu: int) -> torch.device:

    device = torch.device('cuda:%d'%(int(gpu)))
    torch.cuda.set_device(device)
    cfg.device = gpu
    cfg.is_main_process = (gpu == 0)

    if cfg.use_ddp:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5678'
        
        torch.distributed.init_process_group(backend="nccl", world_size = len(cfg.devices), rank=gpu)

    return device


if __name__ == "__main__":
    main()