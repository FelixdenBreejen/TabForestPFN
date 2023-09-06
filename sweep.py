from __future__ import annotations

import os
import time
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import itertools
import pandas as pd
import subprocess
from pathlib import Path
import torch.multiprocessing as mp

from tabularbench.data.benchmarks import benchmarks, benchmark_names
from tabularbench.sweeps.monitor_and_make_plots import monitor_and_make_plots
from tabularbench.sweeps.run_sweeps import run_sweeps
from tabularbench.sweeps.paths_and_filenames import SWEEP_FILE_NAME, PATH_TO_ALL_BENCH_CSV, CONFIG_DUPLICATE
from tabularbench.sweeps.sweep_config import SweepConfig, save_sweep_config_list_to_file


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    if cfg.continue_last_output:
        delete_current_output_dir(cfg)
        set_config_dir_to_last_output(cfg)

    check_existence_of_benchmark_results_csv()
    create_sweep_csv(cfg)
    save_config(cfg)
    launch_sweeps(cfg)


def set_config_dir_to_last_output(cfg: DictConfig) -> None:

    all_output_dirs = list(Path('outputs').glob('*/*'))
    newest_output_dir = max(all_output_dirs, key=output_dir_to_date)
    cfg.output_dir = newest_output_dir


def output_dir_to_date(output_dir: Path) -> datetime:
    
    parts = output_dir.parts
    time_str = "-".join(parts[1:])
    date = datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S")

    return date


def delete_current_output_dir(cfg: DictConfig) -> None:

    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        subprocess.run(['rm', '-rf', output_dir])


def create_sweep_csv(cfg: dict) -> None:

    sweep_configs = []

    assert len(cfg.models) == len(cfg.model_plot_names), f"Please provide a plot name for each model. Got {len(cfg.models)} models and {len(cfg.model_plot_names)} plot names."

    models_with_plot_name = zip(cfg.models, cfg.model_plot_names)
    sweep_details = itertools.product(models_with_plot_name, cfg.search_type, cfg.benchmarks)

    for (model, model_plot_name), search_type, benchmark_name in sweep_details:

        benchmark = benchmarks[benchmark_name]

        if search_type == 'default':
            runs_per_dataset = 1
        else:
            runs_per_dataset = cfg.runs_per_dataset

        if benchmark['categorical']:
            feature_type = 'mixed'
        else:
            feature_type = 'numerical'
            

        sweep_config = SweepConfig(
            model=model,
            plot_name=model_plot_name,
            benchmark_name=benchmark['name'],
            search_type=search_type,
            task=benchmark['task'],
            dataset_size=benchmark['dataset_size'],
            feature_type=feature_type,
            suite_id=benchmark['suite_id'],
            runs_per_dataset=runs_per_dataset
        )

        sweep_configs.append(sweep_config)

    save_sweep_config_list_to_file(sweep_configs, Path(cfg.output_dir) / SWEEP_FILE_NAME)


def save_config(cfg: DictConfig) -> None:
    
    config_path = Path(cfg.output_dir) / CONFIG_DUPLICATE
    OmegaConf.save(cfg, config_path)


def launch_sweeps(cfg) -> None:

    gpus = list(cfg.devices) * cfg.runs_per_device
    path = cfg.output_dir

    time_seed = int(time.time()) * cfg.continue_last_output
    mp.set_start_method('spawn')

    processes = []
    for seed, gpu in enumerate(gpus):
        
        seed += time_seed
        process = mp.Process(target=run_sweeps, args=(path, gpu, seed), )
        process.start()
        processes.append(process)

    print(f"Launched {len(gpus)} agents on {len(set(gpus))} devices")

    monitor_and_make_plots(path, cfg.monitor_interval_in_seconds)

    for process in processes:
        process.join()



def check_existence_of_benchmark_results_csv() -> None:

    results_csv = Path(PATH_TO_ALL_BENCH_CSV)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please download it from the link in the README.")


if __name__ == "__main__":
    main()