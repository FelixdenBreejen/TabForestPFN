from __future__ import annotations

import os
import time
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import argparse
import pandas as pd
import subprocess
from pathlib import Path
import multiprocessing as mp

from tabularbench.launch_benchmarks.launch_benchmarks import benchmarks
from tabularbench.sweeps.monitor_and_make_plots import monitor_and_make_plots
from tabularbench.sweeps.run_sweeps import run_sweeps
from tabularbench.sweeps.paths_and_filenames import SWEEP_FILE_NAME, PATH_TO_ALL_BENCH_CSV


@hydra.main(version_base=None, config_path="config", config_name="sweep")
def main(cfg: DictConfig):

    if cfg.continue_last_output:
        delete_current_output_dir(cfg)
        set_config_dir_to_last_output(cfg)

    check_for_benchmark_results_csv()
    create_sweep_csv(cfg)
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

    processes = []
    for seed, gpu in enumerate(gpus):
        process = mp.Process(target=run_sweeps, args=(path, gpu, seed), )
        process.start()
        processes.append(process)

    print(f"Launched {len(gpus)} agents on {len(set(gpus))} devices")

    monitor_and_make_plots(path, cfg.monitor_interval_in_seconds)

    for process in processes:
        process.join()



def check_for_benchmark_results_csv() -> None:

    results_csv = Path(PATH_TO_ALL_BENCH_CSV)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please download it from the link in the README.")

if __name__ == "__main__":
    main()