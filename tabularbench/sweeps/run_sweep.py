from __future__ import annotations
from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp

from tabularbench.core.enums import SearchType
from tabularbench.results.run_metrics import RunMetrics
from tabularbench.results.run_results import RunResults
from tabularbench.sweeps.config_dataset_sweep import ConfigDatasetSweep
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.sweep_config import SweepConfig
from tabularbench.sweeps.paths_and_filenames import RESULTS_FILE_NAME
from tabularbench.sweeps.config_run import ConfigRun
from tabularbench.sweeps.run_experiment import run_experiment


def run_sweep(cfg: ConfigDatasetSweep):

    cfg.logger.info(f"Start {cfg.search_type.value} search for {cfg.model_name.value} on openml dataset {cfg.openml_dataset_name} ({cfg.openml_dataset_id})")

    results_path = cfg.output_dir / RESULTS_FILE_NAME
    runs_per_dataset = cfg.n_random_runs if cfg.search_type == SearchType.RANDOM else 1

    manager = mp.Manager()
    gpu_queue = manager.Queue()
    run_results_list = manager.list()

    for device in cfg.devices:
        gpu_queue.put(device)

    gpu = gpu_queue.get()
    processes = []
    while len(run_results_list) < runs_per_dataset:
        p = mp.Process(target=run_a_run, args=(cfg, gpu, gpu_queue, run_results_list))
        p.start()
        processes.append(p)
        gpu = gpu_queue.get()

    for p in processes:
        p.join()
    
    for result in run_results_list:
        print(result)

    cfg.logger.info(f"Finished {cfg.search_type.name} search for")


def run_a_run(cfg: ConfigDatasetSweep, device: torch.device, device_queue: mp.Queue, run_result_list: mp.list):

    hyperparam_drawer = HyperparameterDrawer(cfg.hyperparams_object)
    hyperparams = hyperparam_drawer.draw_config(cfg.search_type)
    config_run = ConfigRun.create(cfg, device, hyperparams)
    metrics = run_experiment(config_run)

    if metrics is None:
        cfg.logger.info(f"Run crashed for {cfg.model_name.value} on {cfg.openml_dataset_name} with dataset {cfg.openml_dataset_id}")
        device_queue.put(device)
        return

    run_result = RunResults.from_run_config(config_run, cfg.search_type, metrics)
    run_result_list.append(run_result)
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
