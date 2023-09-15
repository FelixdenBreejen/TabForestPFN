from __future__ import annotations
import argparse
from pathlib import Path
import random
import sys
import yaml
from omegaconf import OmegaConf, DictConfig
import logging

import pandas as pd
import torch.multiprocessing as mp

from tabularbench.configs.all_model_configs import total_config
from tabularbench.core.enums import SearchType
from tabularbench.run_experiment import train_model_on_config
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.sweep_config import SweepConfig, create_sweep_config_list_from_main_config
from tabularbench.sweeps.datasets import get_unfinished_dataset_ids
from tabularbench.sweeps.paths_and_filenames import SWEEP_FILE_NAME, RESULTS_FILE_NAME, CONFIG_DUPLICATE
from tabularbench.sweeps.run_config import RunConfig
from tabularbench.sweeps.sweep_start import get_config, get_logger, set_seed, add_device_to_cfg
from tabularbench.sweeps.writer import Writer
from tabularbench.sweeps.run_experiment import run_experiment


def run_sweeps(output_dir: str, writer_queue: mp.Queue, gpu: int, seed: int = 0):
    """
    Run all sweeps as specified in the config file.

    Args:
        output_dir: Path to the output directory
        writer_queue: Queue for writing to files to prevent race conditions
        gpu: GPU number
        seed: Seed
    """

    cfg = get_config(output_dir)

    logger = get_logger(cfg, log_file_name="gpu_{gpu}_seed_{seed}.log")
    writer = Writer(writer_queue)

    add_device_to_cfg(cfg, gpu)

    logger.info(f"Run sweeps started: device {cfg['device']}, seed {seed}")

    sweep_configs = create_sweep_config_list_from_main_config(cfg, writer, logger)

    logger.info(f"Found {len(sweep_configs)} sweeps to execute")

    for sweep_config in sweep_configs: 
        
        logger.info(f"Sweep config: benchmark {sweep_config.benchmark_name}, model {sweep_config.model}, search type {sweep_config.search_type}") 
              
        search_sweep(sweep_config, SearchType.DEFAULT)
        if sweep_config.search_type == SearchType.RANDOM:
            search_sweep(sweep_config, SearchType.RANDOM)




def search_sweep(sweep: SweepConfig, search_type: SearchType):
    """Perform one sweep: one row of the sweep.csv file."""

    sweep.logger.info(f"Start {sweep.search_type} search for {sweep.model} on {sweep.task}, search type {search_type}")
    set_seed(sweep.seed)

    hyperparam_drawer = HyperparameterDrawer(sweep.hyperparams)
    results_path = sweep.output_dir / RESULTS_FILE_NAME
    runs_per_dataset = sweep.runs_per_dataset if search_type == SearchType.RANDOM else 1
    
    while True:

        datasets_unfinished = get_unfinished_dataset_ids(sweep.openml_dataset_ids, results_path, runs_per_dataset)

        if len(datasets_unfinished) == 0:
            break

        hyperparams = hyperparam_drawer.draw_config(search_type)
        dataset_id = random.choice(datasets_unfinished)

        config_run = RunConfig.create(sweep, dataset_id, hyperparams)
        results = run_experiment(config_run)

        if results is None:
            # This is the error code in case the run crashes
            continue

        if config_run.openml_dataset_id not in get_unfinished_dataset_ids(sweep.openml_dataset_ids, results_path, runs_per_dataset):
            # This is the case where another process finished the dataset while this process was running
            # It is important to check this because otherwise the results default runs will be saved multiple times,
            # which is problematic for computing random search statistics.
            continue

        save_results(config_run, results, results_path)

    

def save_results(config_run: RunConfig, results: dict, results_path: Path):

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

