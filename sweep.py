import hydra
from omegaconf import DictConfig, OmegaConf

from tabularbench.run_experiment import train_model_on_config



@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['debug'] = False
    
    

    train_model_on_config(cfg_dict)


if __name__ == "__main__":
    main()