import hydra
from omegaconf import DictConfig, OmegaConf

import random
import numpy as np
import torch

from tabularbench.core.trainer_masked_saint import TrainerMaskedSaint


@hydra.main(version_base=None, config_path="config", config_name="pretrain")
def main(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['debug'] = True

    random.seed(cfg_dict['seed'])
    np.random.seed(cfg_dict['seed'])
    torch.manual_seed(cfg_dict['seed'])
    
    

    trainer = TrainerMaskedSaint(cfg_dict)
    trainer.train()


if __name__ == "__main__":
    main()