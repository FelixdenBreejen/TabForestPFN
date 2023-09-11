from __future__ import annotations
import argparse
from pathlib import Path
import os
import sys
import yaml

import pandas as pd
import random
import numpy as np
import torch
import torch.multiprocessing as mp

from tabularbench.configs.all_model_configs import total_config
from tabularbench.run_experiment import train_model_on_config
from tabularbench.sweeps.random_search_object import WandbSearchObject
from tabularbench.sweeps.sweep_config import SweepConfig, load_sweep_configs_from_file
from tabularbench.sweeps.datasets import get_unfinished_task_ids
from tabularbench.sweeps.paths_and_filenames import SWEEP_FILE_NAME, RESULTS_FILE_NAME, CONFIG_DUPLICATE
from tabularbench.sweeps.run_config import RunConfig



def run_sweeps(process_id: int, output_dir: str, writer_queue: mp.Queue, gpu: int, seed: int = 0):
    """
    Run all sweeps as specified in the config file.

    Args:
        output_dir: Path to the output directory
        writer_queue: Queue for writing to files to prevent race conditions
        gpu: GPU number
        seed: Seed
    """

    cfg = get_config(output_dir)

    print("seed: ", seed)

    log_to_file(output_dir)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    with open(Path(output_dir) / CONFIG_DUPLICATE, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg.device = 'cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu'
        cfg.seed = seed

    sweep_csv = pd.read_csv(Path(output_dir) / SWEEP_FILE_NAME)
    sweep_configs = load_sweep_configs_from_file(sweep_csv, output_dir)

    for sweep_config in sweep_configs:  
              
        search_sweep(cfg, sweep_config, is_random=False)
        if sweep_config.search_type == 'random':
            search_sweep(cfg, sweep_config, is_random=True)

    end_log_to_file()


def log_to_file(output_dir: str):

    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name_stdout = str(os.getpid()) + ".out"
    log_file_name_stderr = str(os.getpid()) + "_error.out"

    sys.stdout = open(log_dir / log_file_name_stdout, "a", 1)
    sys.stderr = open(log_dir / log_file_name_stderr, "a", 1)


def end_log_to_file():
    
    sys.stdout.close()
    sys.stderr.close()


def search_sweep(cfg: dict, sweep: SweepConfig, is_random: bool):
    """Perform one sweep: one row of the sweep.csv file."""
    

    assert sweep.model in total_config.keys(), f"Model {sweep.model} not found in total_config"

    config = total_config[sweep.model][sweep.task]
    search_object = WandbSearchObject(config)
    results_path = sweep.path / RESULTS_FILE_NAME
    runs_per_dataset = sweep.runs_per_dataset if is_random else 1
    
    while True:

        run_config = RunConfig.from_cfg_and_sweep_cfg(cfg, sweep, is_random)

        datasets_unfinished = get_unfinished_task_ids(sweep.task_ids, results_path, runs_per_dataset)

        if len(datasets_unfinished) == 0:
            break
        
        config_run = create_run_config(cfg, sweep, datasets_unfinished, search_object, is_random)
        results = train_model_on_config(config_run)

        if results == -1:
            # This is the error code in case the run crashes
            continue

        if config_run['data__keyword'] not in get_unfinished_task_ids(sweep.task_ids, results_path, runs_per_dataset):
            # This is the case where another process finished the dataset while this process was running
            # It is important to check this because otherwise the results default runs will be saved multiple times,
            # which is problematic for computing random search statistics.
            continue

        save_results(results, results_path)

    

def save_results(results: dict, results_path: Path):

    df_new = pd.Series(results).to_frame().T

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(results_path, mode='w', index=False, header=True)
    else:
        df = pd.read_csv(results_path)
        df = df.append(df_new, ignore_index=True)
        df.to_csv(results_path, mode='w', index=False, header=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run random search sweeps')
    parser.add_argument('--output_dir', type=str, help='Path to sweep output directory')
    parser.add_argument('--seed', type=int, help='Seed')
    parser.add_argument('--main_process', action='store_true', help='Whether this is the main process (makes graphs)')

    args = parser.parse_args()

    run_sweeps(args.output_dir, seed=args.seed, main_process=args.main_process)

