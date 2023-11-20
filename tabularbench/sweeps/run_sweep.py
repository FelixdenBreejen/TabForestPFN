from __future__ import annotations
from pathlib import Path
import random

import pandas as pd
import torch
import torch.multiprocessing as mp

from tabularbench.core.enums import SearchType
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.results.run_results import RunResults
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.paths_and_filenames import RESULTS_FILE_NAME
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.sweeps.run_experiment import run_experiment
from tabularbench.sweeps.sweep_start import get_logger


def run_sweep(cfg: ConfigBenchmarkSweep):

    cfg.logger.info(f"Start {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")

    results_path = cfg.output_dir / RESULTS_FILE_NAME
    runs_per_dataset = cfg.n_random_runs if cfg.search_type == SearchType.RANDOM else 1

    manager = mp.Manager()
    device_queue = manager.Queue()
    run_results_dict = manager.dict()
    runs_busy_dict = manager.dict()
    runs_attempted_dict = {}

    for dataset_id in cfg.benchmark.openml_dataset_ids:
        run_results_dict[dataset_id] = manager.list()
        runs_busy_dict[dataset_id] = 0
        runs_attempted_dict[dataset_id] = 0

    for device in cfg.devices:
        device_queue.put(device)

    device = device_queue.get()

    hyperparam_drawer = HyperparameterDrawer(cfg.hyperparams_object)

    while not all_runs_finished(run_results_dict, runs_per_dataset):

        if all_runs_almost_finished(run_results_dict, runs_per_dataset, runs_busy_dict):
            # The last few runs are being executed, so we need to wait for them to finish
            # These last runs might have an error, so we need to be able to redo them if necessary
            cfg.logger.info(f"Waiting for last {sum(runs_busy_dict.values())} runs to finish...")
            device = device_queue.get()
            continue

        cfg.logger.info(f"Preparing a new run")
        cfg.logger.info(f"Currently, {sum(runs_busy_dict.values())} runs are busy and {sum(runs_attempted_dict.values())} runs have been attempted")

        hyperparams = hyperparam_drawer.draw_config(cfg.search_type)
        dataset_id = draw_dataset_id(cfg.benchmark.openml_dataset_ids, run_results_dict, runs_per_dataset, runs_busy_dict)
        config_run = ConfigRun.create(cfg, device, dataset_id, hyperparams, runs_attempted_dict[dataset_id])

        runs_busy_dict[dataset_id] += 1
        runs_attempted_dict[dataset_id] += 1

        cfg.logger.info(f"Start {cfg.search_type.value} run for {cfg.model_name.value} on {cfg.benchmark.name} with dataset {config_run.openml_dataset_id} (id={config_run.openml_dataset_id})")

        mp.Process(target=run_a_run, args=(config_run, device, device_queue, run_results_dict, runs_busy_dict, cfg.search_type)).start()
        device = device_queue.get()   # blocks until a gpu is available
        cfg.logger.info(f"A free device {device} is found and grabbed")

    cfg.logger.info(f"Finished {cfg.search_type.name} search for {cfg.model_name.name} on {cfg.benchmark.name}")


def all_runs_finished(run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int) -> bool:
    # All runs are finished if there are no datasets left with less than runs_per_dataset runs

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) < runs_per_dataset:
            return False
    else:
        return True


def all_runs_almost_finished(run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int, runs_busy_dict: dict[int, int]) -> bool:
    # All runs are almost finished if there are no datasets left if all runs that are currently running are finished

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) + runs_busy_dict[dataset_id] < runs_per_dataset:
            return False
    else:
        return True


def draw_dataset_id(openml_dataset_ids: list[int], run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int, runs_busy_dict: dict[int, int]) -> int:
    # We draw multinomially from the number of runs left for each dataset
    
    runs_left = [runs_per_dataset - len(run_results_dict[dataset_id]) - runs_busy_dict[dataset_id] for dataset_id in openml_dataset_ids]
    dataset_id = random.choices(openml_dataset_ids, runs_left, k=1)[0]
    return dataset_id


def run_a_run(
        cfg: ConfigRun, 
        device: torch.device, 
        device_queue: mp.Queue, 
        run_results_dict: dict[int, list[RunResults]], 
        runs_busy_dict: dict[int, int], 
        search_type: SearchType
    ):

    # logger needs to be reinitialized because of multiprocessing issues
    cfg.logger = get_logger(cfg.output_dir / 'log.txt')

    metrics = run_experiment(cfg)

    if metrics is None:
        cfg.logger.info(f"Run crashed for {cfg.model_name.value} on {cfg.openml_dataset_name} with dataset {cfg.openml_dataset_id}")
        device_queue.put(device)
        return

    run_result = RunResults.from_run_config(cfg, search_type, metrics)
    run_results_dict[cfg.openml_dataset_id].append(run_result)
    runs_busy_dict[cfg.openml_dataset_id] -= 1
    device_queue.put(device)




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
