from __future__ import annotations

import os
import time
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import pandas as pd
import subprocess
from pathlib import Path

from tabularbench.run_experiment import train_model_on_config
from tabularbench.launch_benchmarks.launch_benchmarks import benchmarks
from tabularbench.launch_benchmarks.launch_benchmarks import main as make_wandb_sweeps
from tabularbench.launch_benchmarks.monitor import main as monitor_sweeps
from tabularbench.run_sweeps import run_sweeps, SWEEP_FILE_NAME


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    create_sweep_csv(cfg)
    launch_sweeps(cfg)


def create_sweep_csv(cfg: dict) -> None:

    sweep_dicts = []

    for i_model, model in enumerate(cfg.models):
        for task in cfg.random_search:
            for benchmark in benchmarks:

                if benchmark['name'] not in cfg.benchmarks:
                    continue
                
                sweep_dict = {
                    'model': model,
                    'plot_name': cfg.model_plot_names[i_model], 
                    'benchmark': benchmark['name'],
                    'random_search': task,
                    'task': benchmark['task'],
                    'dataset_size': benchmark['dataset_size'],
                    'categorical': benchmark['categorical'],
                    'suite_id': benchmark['suite_id'],
                    'runs_per_dataset': 1 if task == 'default' else cfg.runs_per_dataset
                }

                sweep_dicts.append(sweep_dict)

    sweep_csv_path = os.path.join(cfg.output_dir, SWEEP_FILE_NAME)
    pd.DataFrame(sweep_dicts).to_csv(sweep_csv_path, index=False)


def launch_sweeps(cfg) -> None:

    gpus = list(cfg.devices) * cfg.runs_per_device
    path = cfg.output_dir

    if len(gpus) > 1:
        for i, gpu in enumerate(gpus[1:]):
            subprocess.run(['bash', 'tabularbench/launch_benchmarks/launch_agent_tmux.sh', '-g', str(gpu), '-p', path, '-s', str(i+1)])
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[0])
    print(f"Launched {len(gpus)} agents on {len(set(gpus))} devices")

    run_sweeps(path, main_process=True)

if __name__ == "__main__":
    main()