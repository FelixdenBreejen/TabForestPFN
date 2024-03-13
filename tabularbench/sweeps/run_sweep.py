from __future__ import annotations

import random

import torch
import torch.multiprocessing as mp
from loguru import logger

from tabularbench.core.enums import SearchType
from tabularbench.core.run_experiment import run_experiment
from tabularbench.data.datafile_name_maker import make_datafile_path
from tabularbench.results.results_run import ResultsRun
from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.sweeps.hyperparameter_drawer import HyperparameterDrawer
from tabularbench.sweeps.make_plots import plot_results
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.config_run import ConfigRun
from tabularbench.utils.paths_and_filenames import RESULTS_FILE_NAME


def run_sweep(cfg: ConfigBenchmarkSweep):

    logger.info(f"Start {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    cfg.save()

    log_ignore_datasets(cfg)

    sweep_runner = SweepRunner(cfg)
    sweep_runner.main_loop()

    logger.info(f"Finished {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")



class SweepRunner():

    def __init__(self, cfg: ConfigBenchmarkSweep):

        self.cfg = cfg
        self.runs_per_dataset = self.get_runs_per_dataset()

        self.manager = mp.Manager()
        self.device_queue = self.manager.Queue()
        self.results_run_dict = self.manager.dict()
        self.runs_busy_dict = self.manager.dict()
        self.runs_attempted_dict = {}

        self.initalize_run_dicts()
        self.initalize_device_queue()

        self.hyperparam_drawer = HyperparameterDrawer(cfg.hyperparams_object)


    def initalize_run_dicts(self) -> None:

        for dataset_id in self.cfg.benchmark.openml_dataset_ids:
            self.results_run_dict[dataset_id] = self.manager.list()
            self.runs_busy_dict[dataset_id] = 0
            self.runs_attempted_dict[dataset_id] = 0


    def initalize_device_queue(self) -> None:

        for device in self.cfg.devices:
            self.device_queue.put(device)

        
    @property
    def n_runs_busy(self) -> int:
        return sum(self.runs_busy_dict.values())
    
    @property
    def n_runs_attempted(self) -> int:
        return sum(self.runs_attempted_dict.values())


    def main_loop(self) -> None:

        device = self.device_queue.get()

        while not self.all_runs_finished():

            if self.all_runs_almost_finished():
                # The last few runs are being executed, so we need to wait for them to finish
                # These last runs might have an error, so we need to be able to redo them if necessary
                logger.info(f"Waiting for last {self.n_runs_busy} run(s) to finish...")
                device = self.device_queue.get()   # blocks until a gpu is available (run is finished)
                self.process_results()
                continue

            logger.info(f"Currently, {self.n_runs_busy} runs are busy and {self.n_runs_attempted} runs have been attempted")

            dataset_id = self.draw_dataset_id()
            hyperparam_search_type, seed = self.get_hyperparam_search_type_and_seed(dataset_id)
            hyperparams = self.hyperparam_drawer.draw_config(hyperparam_search_type)

            config_run = ConfigRun.create(
                cfg=self.cfg, 
                seed=seed, 
                device=device, 
                dataset_file_path=make_datafile_path(self.cfg.benchmark.origin, dataset_id, self.cfg.benchmark.dataset_size),
                hyperparams=hyperparams, 
                run_id=self.runs_attempted_dict[dataset_id]
            )

            self.runs_busy_dict[dataset_id] += 1
            self.runs_attempted_dict[dataset_id] += 1

            logger.info(f"Start {self.cfg.search_type.value} run for {self.cfg.model_name.value} on {self.cfg.benchmark.name} with dataset {config_run.openml_dataset_name} (id={config_run.openml_dataset_id})")

            mp.Process(target=run_a_run, args=(config_run, device, self.device_queue, self.results_run_dict, self.runs_busy_dict, hyperparam_search_type)).start()
        
            self.process_results()
            device = self.device_queue.get()   # blocks until a gpu is available

            logger.info(f"A free device {device} is found and grabbed")


        self.process_results()


    def no_runs_finished(self) -> bool:
        return all([len(self.results_run_dict[dataset_id]) == 0 for dataset_id in self.results_run_dict])


    def all_runs_finished(self) -> bool:
        # All runs are finished if there are no datasets left with less than runs_per_dataset runs

        for dataset_id in self.results_run_dict.keys():
            if len(self.results_run_dict[dataset_id]) < self.runs_per_dataset and dataset_id not in self.cfg.openml_dataset_ids_to_ignore:
                return False
        else:
            return True


    def all_runs_almost_finished(self) -> bool:
        # All runs are almost finished if there are no datasets left if all runs that are currently running are finished

        for dataset_id in self.results_run_dict.keys():
            if len(self.results_run_dict[dataset_id]) + self.runs_busy_dict[dataset_id] < self.runs_per_dataset and dataset_id not in self.cfg.openml_dataset_ids_to_ignore:
                return False
        else:
            return True


    def get_runs_per_dataset(self) -> int:

        match self.cfg.search_type:
            case SearchType.RANDOM:
                runs_per_dataset = self.cfg.n_random_runs_per_dataset
            case SearchType.DEFAULT:
                runs_per_dataset = self.cfg.n_default_runs_per_dataset
                
        return runs_per_dataset


    def process_results(self) -> None:

        if self.no_runs_finished():
            return
        
        results_run_dict = {dataset_id: list(results_runs) for dataset_id, results_runs in self.results_run_dict.items()}
        results_sweep = ResultsSweep.from_run_results_dict(results_run_dict)
        results_sweep.save(self.cfg.output_dir / RESULTS_FILE_NAME)
        plot_results(self.cfg, results_sweep)


    def draw_dataset_id(self) -> int:
        # We draw multinomially from the number of runs left for each dataset
        
        openml_dataset_ids = self.cfg.benchmark.openml_dataset_ids
        banned_dataset_ids = self.cfg.openml_dataset_ids_to_ignore

        runs_left = [self.runs_per_dataset - len(self.results_run_dict[dataset_id]) - self.runs_busy_dict[dataset_id] for dataset_id in openml_dataset_ids]
        
        for dataset_id in banned_dataset_ids:
            runs_left[openml_dataset_ids.index(dataset_id)] = 0
        
        dataset_id = random.choices(openml_dataset_ids, runs_left, k=1)[0]
        return dataset_id
    

    def get_hyperparam_search_type_and_seed(self, dataset_id: int) -> tuple[SearchType, int]:

        if len(self.results_run_dict[dataset_id]) == 0 and self.runs_busy_dict[dataset_id] == 0:
            # This is the first run for this dataset, so we draw the default hyperparams
            return SearchType.DEFAULT, self.cfg.seed
        elif self.cfg.search_type == SearchType.DEFAULT:
            return SearchType.DEFAULT, self.cfg.seed + self.runs_attempted_dict[dataset_id]
        else:
            return SearchType.RANDOM, self.cfg.seed


def run_a_run(
    cfg: ConfigRun, 
    device: torch.device, 
    device_queue: mp.Queue, 
    run_results_dict: dict[int, list[ResultsRun]], 
    runs_busy_dict: dict[int, int], 
    search_type: SearchType
):


    logger.add(cfg.output_dir / "log.log", enqueue=True)
    metrics = run_experiment(cfg)

    if metrics is None:
        logger.info(f"Run crashed for {cfg.model_name.value} on {cfg.openml_dataset_name} with dataset {cfg.openml_dataset_id}")
        device_queue.put(device)
        return

    run_result = ResultsRun.from_run_config(cfg, search_type, metrics)
    run_results_dict[cfg.openml_dataset_id].append(run_result)
    runs_busy_dict[cfg.openml_dataset_id] -= 1
    device_queue.put(device)


def log_ignore_datasets(cfg: ConfigBenchmarkSweep) -> None:

    if len(cfg.openml_dataset_ids_to_ignore) > 0:
        logger.info("The following openml datasets will be ignored:")
        for dataset_id in cfg.openml_dataset_ids_to_ignore:
            dataset_name = cfg.benchmark.openml_dataset_names[cfg.benchmark.openml_dataset_ids.index(dataset_id)]
            logger.info(f"    {dataset_name} (id={dataset_id})")
    else:
        logger.info("All openml datasets in the benchmark will be used, non ignored")
