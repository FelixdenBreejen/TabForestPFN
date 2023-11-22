from __future__ import annotations

import hydra
from omegaconf import DictConfig
import torch.multiprocessing as mp
from main import check_existence_of_benchmark_results_csv, save_config
from tabularbench.sweeps.config_pretrain import ConfigPretrain

from tabularbench.sweeps.set_seed import set_seed



@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(cfg_hydra: DictConfig):
    
    mp.set_start_method('spawn')

    cfg = ConfigPretrain.from_hydra(cfg_hydra)
    cfg.logger.info("Finished creating pretrain config")

    check_existence_of_benchmark_results_csv(cfg)
    save_config(cfg)
    set_seed(cfg.seed)
    
    trainer = TrainerMaskedSaint(cfg_dict)
    trainer.train()

if __name__ == "__main__":
    main()