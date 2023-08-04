import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import pandas as pd
import subprocess
from pathlib import Path

from tabularbench.run_experiment import train_model_on_config
from tabularbench.launch_benchmarks.launch_benchmarks import main as make_wandb_sweeps
from tabularbench.launch_benchmarks.monitor import main as monitor_sweeps

# os.environ["WANDB_MODE"] = "offline"

@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    sweep_df_path = os.path.join(cfg.output_dir, 'wandb_sweep.csv') 

    launch_benchmarks_args = argparse.Namespace(**{
        'benchmarks': [cfg.benchmark],
        'models': [cfg.model],
        'output_file': sweep_df_path,
        'datasets': [],
        'exclude': [],
        'suffix': '',
        'default': [not cfg.random_search]
    })

    sweep_df = make_wandb_sweeps(launch_benchmarks_args)

    launch_sweeps(cfg, sweep_df)

    results_path = os.path.join(cfg.output_dir, 'results.csv')

    monitor_sweeps_args = argparse.Namespace(**{
        'filename': sweep_df_path,
        'max_runs': 2,                                         # Max runs per dataset
        'output_filename': results_path,
        'default': not cfg.random_search,
        'max_run_per_sweep': 20000,
        'time': 10
    })

    monitor_sweeps(monitor_sweeps_args)

    pass


def launch_sweeps(cfg, sweep_df: pd.DataFrame) -> None:

    sweep_id = sweep_df.iloc[0]['sweep_id']
    project = sweep_df.iloc[0]['project']

    gpus = list(cfg.devices) * cfg.runs_per_device

    for gpu in gpus:
        subprocess.run(['bash', 'tabularbench/launch_benchmarks/launch_agent_tmux.sh', '-g', str(gpu), '-p', project, '-s', sweep_id])

    print(f"Launched {len(gpus)} agents on {len(set(gpus))} devices")

    



def run_agents(gpu:int, sweep_df: pd.DataFrame):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i, row in sweep_df.iterrows():
        os.system(f'wandb agent {row["project"]}/{row["sweep_id"]}')



if __name__ == "__main__":
    main()