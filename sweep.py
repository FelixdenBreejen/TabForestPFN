import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import pandas as pd
import multiprocessing as mp
from pathlib import Path

from tabularbench.run_experiment import train_model_on_config
from tabularbench.launch_benchmarks.launch_benchmarks import main as make_wandb_sweeps



@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    
    launch_benchmarks_args = argparse.Namespace(**{
        'benchmarks': cfg.benchmarks,
        'models': cfg.models,
        'output_file': os.path.join(cfg.output_dir, 'wandb_sweep.csv'),
        'datasets': [],
        'exclude': [],
        'suffix': '',
        'default': [~a for a in cfg.random_search]
    })

    sweep_df = make_wandb_sweeps(launch_benchmarks_args)

    launch_sweeps(cfg, sweep_df)



    train_model_on_config(cfg_dict)


def launch_sweeps(cfg, sweep_df: pd.DataFrame) -> None:

    gpus = list(cfg.devices) * cfg.runs_per_device

    for gpu in gpus:

        p = mp.Process(target = run_agents, args=(gpu, sweep_df))
        p.start()

    



def run_agents(gpu:int, sweep_df: pd.DataFrame):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i, row in sweep_df.iterrows():
        os.system(f'wandb agent {row["project"]}/{row["sweep_id"]}')

    print('All done!')



if __name__ == "__main__":
    main()