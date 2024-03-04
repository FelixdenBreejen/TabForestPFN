from __future__ import annotations
import random

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger

from tabularbench.core.enums import SearchType
from tabularbench.data.datafile_name_maker import make_datafile_path
from tabularbench.results.run_results import RunResults
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.make_plots import plot_results
from tabularbench.utils.paths_and_filenames import RESULTS_FILE_NAME
from tabularbench.utils.config_run import ConfigRun
from tabularbench.core.run_experiment import run_experiment


def run_sweep(cfg: ConfigBenchmarkSweep):

    logger.info(f"Start {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    cfg.save()

    log_ignore_datasets(cfg)

    match cfg.search_type:
        case SearchType.RANDOM:
            runs_per_dataset = cfg.n_random_runs_per_dataset
        case SearchType.DEFAULT:
            runs_per_dataset = cfg.n_default_runs_per_dataset

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

    while not all_runs_finished(cfg, run_results_dict, runs_per_dataset):

        if all_runs_almost_finished(cfg, run_results_dict, runs_per_dataset, runs_busy_dict):
            # The last few runs are being executed, so we need to wait for them to finish
            # These last runs might have an error, so we need to be able to redo them if necessary
            logger.info(f"Waiting for last {sum(runs_busy_dict.values())} run(s) to finish...")
            device = device_queue.get()   # blocks until a gpu is available (run is finished)
            run_results_df = convert_run_results_dict_to_dataframe(run_results_dict)
            save_results(cfg, run_results_df)
            plot_results(cfg, run_results_df)
            continue

        logger.info(f"Currently, {sum(runs_busy_dict.values())} runs are busy and {sum(runs_attempted_dict.values())} runs have been attempted")

        dataset_id = draw_dataset_id(cfg, run_results_dict, runs_per_dataset, runs_busy_dict)

        hyperparam_search_type = cfg.search_type
        if len(run_results_dict[dataset_id]) == 0 and runs_busy_dict[dataset_id] == 0:
            # This is the first run for this dataset, so we draw the default hyperparams
            hyperparam_search_type = SearchType.DEFAULT
            seed = cfg.seed
        elif cfg.search_type == SearchType.DEFAULT:
            seed = cfg.seed + runs_attempted_dict[dataset_id]
            
        hyperparams = hyperparam_drawer.draw_config(hyperparam_search_type)
        config_run = ConfigRun.create(
            cfg=cfg, 
            seed=seed, 
            device=device, 
            dataset_file_path=make_datafile_path(cfg.benchmark.origin, dataset_id, cfg.benchmark.dataset_size),
            hyperparams=hyperparams, 
            run_id=runs_attempted_dict[dataset_id]
        )

        runs_busy_dict[dataset_id] += 1
        runs_attempted_dict[dataset_id] += 1

        logger.info(f"Start {cfg.search_type.value} run for {cfg.model_name.value} on {cfg.benchmark.name} with dataset {config_run.openml_dataset_name} (id={config_run.openml_dataset_id})")

        mp.Process(target=run_a_run, args=(config_run, device, device_queue, run_results_dict, runs_busy_dict, hyperparam_search_type)).start()
       
        run_results_df = convert_run_results_dict_to_dataframe(run_results_dict)
        save_results(cfg, run_results_df)
        plot_results(cfg, run_results_df)

        device = device_queue.get()   # blocks until a gpu is available
        logger.info(f"A free device {device} is found and grabbed")


    logger.info(f"Finished {cfg.search_type.name} search for {cfg.model_name.name} on {cfg.benchmark.name}")



def log_ignore_datasets(cfg: ConfigBenchmarkSweep) -> None:

    if len(cfg.openml_dataset_ids_to_ignore) > 0:
        logger.info("The following openml datasets will be ignored:")
        for dataset_id in cfg.openml_dataset_ids_to_ignore:
            dataset_name = cfg.benchmark.openml_dataset_names[cfg.benchmark.openml_dataset_ids.index(dataset_id)]
            logger.info(f"    {dataset_name} (id={dataset_id})")
    else:
        logger.info("All openml datasets in the benchmark will be used, non ignored")


def all_runs_finished(cfg: ConfigRun, run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int) -> bool:
    # All runs are finished if there are no datasets left with less than runs_per_dataset runs

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) < runs_per_dataset and dataset_id not in cfg.openml_dataset_ids_to_ignore:
            return False
    else:
        return True


def all_runs_almost_finished(cfg: ConfigRun, run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int, runs_busy_dict: dict[int, int]) -> bool:
    # All runs are almost finished if there are no datasets left if all runs that are currently running are finished

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) + runs_busy_dict[dataset_id] < runs_per_dataset and dataset_id not in cfg.openml_dataset_ids_to_ignore:
            return False
    else:
        return True


def draw_dataset_id(cfg: ConfigRun, run_results_dict: dict[int, list[RunResults]], runs_per_dataset: int, runs_busy_dict: dict[int, int]) -> int:
    # We draw multinomially from the number of runs left for each dataset
    
    openml_dataset_ids = cfg.benchmark.openml_dataset_ids
    banned_dataset_ids = cfg.openml_dataset_ids_to_ignore

    runs_left = [runs_per_dataset - len(run_results_dict[dataset_id]) - runs_busy_dict[dataset_id] for dataset_id in openml_dataset_ids]
    
    for dataset_id in banned_dataset_ids:
        runs_left[openml_dataset_ids.index(dataset_id)] = 0
    
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


    logger.add(cfg.output_dir / "log.log", enqueue=True)
    metrics = run_experiment(cfg)

    if metrics is None:
        logger.info(f"Run crashed for {cfg.model_name.value} on {cfg.openml_dataset_name} with dataset {cfg.openml_dataset_id}")
        device_queue.put(device)
        return

    run_result = RunResults.from_run_config(cfg, search_type, metrics)
    run_results_dict[cfg.openml_dataset_id].append(run_result)
    runs_busy_dict[cfg.openml_dataset_id] -= 1
    device_queue.put(device)




def save_results(cfg: ConfigBenchmarkSweep, run_results_df: pd.DataFrame) -> None:

    if len(run_results_df) == 0:
        # no results yet to save
        return

    results_path = cfg.output_dir / RESULTS_FILE_NAME
    run_results_df.to_csv(results_path, index=False, header=True)

    logger.info(f"Saved results ({len(run_results_df)} runs total) to {RESULTS_FILE_NAME}")
    

def convert_run_results_dict_to_dataframe(run_results_dict: dict[int, list[RunResults]]) -> pd.DataFrame:

    results = []

    for dataset_id, run_results_list in run_results_dict.items():
        for run_results in run_results_list:
            results.append(run_results.to_dict())

    df = pd.DataFrame(results)
    return df
