from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch.multiprocessing as mp

from tabularbench.core.enums import SearchType
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.results.run_results import RunResults
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.sweep_config import SweepConfig, create_sweep_config_list_from_main_config
from tabularbench.sweeps.datasets import draw_dataset_id, get_unfinished_dataset_ids
from tabularbench.sweeps.paths_and_filenames import RESULTS_FILE_NAME
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.sweeps.sweep_start import get_config, get_logger, set_seed, add_device_and_seed_to_cfg
from tabularbench.sweeps.writer import Writer
from tabularbench.sweeps.run_experiment import run_experiment


def run_sweeps(output_dir: str, writer_queue: mp.JoinableQueue, gpu: int, seed: int = 0):
    """
    Run all sweeps as specified in the config file.

    Args:
        output_dir: Path to the output directory
        writer_queue: Queue for writing to files to prevent race conditions
        gpu: GPU number
        seed: Seed
    """

    cfg = get_config(output_dir)

    logger = get_logger(cfg, log_file_name=f"gpu_{gpu}_seed_{seed}.log")
    writer = Writer(writer_queue)

    add_device_and_seed_to_cfg(cfg, gpu, seed)

    logger.info(f"Run sweeps started: device {cfg['device']}, seed {seed}")

    sweep_configs = create_sweep_config_list_from_main_config(cfg, writer, logger)

    logger.info(f"Found {len(sweep_configs)} sweeps to execute")

    for sweep_config in sweep_configs: 
        
        logger.info(f"Sweep config: benchmark {sweep_config.benchmark_name}, model {sweep_config.model.name}, search type {sweep_config.search_type.name}") 
              
        search_sweep(sweep_config, SearchType.DEFAULT)
        if sweep_config.search_type == SearchType.RANDOM:
            search_sweep(sweep_config, SearchType.RANDOM)




def search_sweep(sweep: SweepConfig, search_type: SearchType):
    """Perform one sweep: one row of the sweep.csv file."""

    sweep.logger.info(f"Start {search_type.name} search for {sweep.model.name} on {sweep.benchmark_name}")
    set_seed(sweep.seed)

    hyperparam_drawer = HyperparameterDrawer(sweep.hyperparams)
    results_path = sweep.sweep_dir / RESULTS_FILE_NAME
    runs_per_dataset = sweep.runs_per_dataset if search_type == SearchType.RANDOM else 1
    
    while True:

        # TODO: because the seed is set everytime a dataset is run, the sweep picks the same dataset and hyperparams over and over again
        datasets_unfinished = get_unfinished_dataset_ids(sweep.openml_dataset_ids, results_path, runs_per_dataset)

        if len(datasets_unfinished) == 0:
            break

        hyperparams = hyperparam_drawer.draw_config(search_type)
        dataset_id = draw_dataset_id(datasets_unfinished, sweep.seed, search_type, first_run=datasets_unfinished == sweep.openml_dataset_ids)

        config_run = ConfigRun.create(sweep, dataset_id, hyperparams)
        metrics = run_experiment(config_run)

        if metrics is None:
            # This is the error code in case the run crashes
            sweep.logger.info(f"Run crashed for {sweep.model.name} on {sweep.benchmark_name} with dataset {dataset_id}")
            continue

        if config_run.openml_dataset_id not in get_unfinished_dataset_ids(sweep.openml_dataset_ids, results_path, runs_per_dataset):
            # This is the case where another process finished the dataset while this process was running
            # It is important to check this because otherwise the results default runs will be saved multiple times,
            # which is problematic for computing random search statistics.
            sweep.logger.info(f"Run finished by another process for {sweep.model.name} on {sweep.benchmark_name} with dataset {dataset_id}")
            sweep.logger.info(f"Results are not being saved.")
            continue

        save_results(sweep, config_run, metrics, results_path, search_type)

    sweep.logger.info(f"Finished {search_type.name} search for {sweep.model.name} on {sweep.benchmark_name}")

    

def save_results(config_sweep: SweepConfig, config_run: ConfigRun, metrics: RunMetrics, results_path: Path, search_type: SearchType):

    results_dict = RunResults.from_run_config(config_run, search_type, metrics).to_dict()

    df_new = pd.Series(results_dict).to_frame().T

    if not results_path.exists():
        results_path.parent.mkdir(parents=True, exist_ok=True)
        csv_string = df_new.to_csv(index=False, header=True)
        config_sweep.writer.write(results_path, csv_string, mode="w")
    else:
        df = pd.read_csv(results_path)
        df = pd.concat([df, df_new], ignore_index=True)
        csv_string = df.to_csv(index=False, header=True)
        config_sweep.writer.write(results_path, csv_string, mode="w")
