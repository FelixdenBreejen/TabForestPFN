from __future__ import annotations

import time
import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import subprocess
from pathlib import Path
import torch.multiprocessing as mp

from tabularbench.sweeps.monitor_and_make_plots import monitor_and_make_plots
from tabularbench.sweeps.run_sweeps import run_sweeps
from tabularbench.sweeps.paths_and_filenames import PATH_TO_ALL_BENCH_CSV, CONFIG_DUPLICATE
from tabularbench.sweeps.writer import file_writer

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sweep.log", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig):

    if cfg.continue_last_output:
        delete_current_output_dir(cfg)
        set_config_dir_to_last_output(cfg)
        logger.info(f"Continuing last output in directory {cfg.output_dir}")

    check_existence_of_benchmark_results_csv()
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



def save_config(cfg: DictConfig) -> None:
    
    config_path = Path(cfg.output_dir) / CONFIG_DUPLICATE

    config_to_save = cfg.copy()

    del config_to_save.devices
    del config_to_save.continue_last_output
    del config_to_save.monitor_interval_in_seconds
    del config_to_save.runs_per_device
    del config_to_save.seed

    OmegaConf.save(config_to_save, config_path, resolve=True)    
    logger.info(f"Saved config to {config_path}")


def launch_sweeps(cfg) -> None:

    gpus = list(cfg.devices) * cfg.runs_per_device
    path = cfg.output_dir

    time_seed = int(time.time()) * cfg.continue_last_output

    mp.set_start_method('spawn')
    writer_queue = mp.JoinableQueue()    # type: ignore

    writing_process = mp.Process(target=file_writer, args=(writer_queue,))
    writing_process.daemon = True
    writing_process.start()
    logger.info(f"Launched writer")

    for gpu_i, gpu in enumerate(gpus):
        sd = cfg.seed + gpu_i + time_seed
        mp.Process(target=run_sweeps, args=(path, writer_queue, gpu, sd)).start()
        logger.info(f"Launched agent on device {gpu} with seed {sd}")

    process = mp.Process(target=monitor_and_make_plots, args=(path, writer_queue, cfg.monitor_interval_in_seconds))
    process.start()
    logger.info(f"Launched monitor and plotter")
    process.join()
    logger.info(f"Monitoring finished, exiting main process")


def check_existence_of_benchmark_results_csv() -> None:

    results_csv = Path(PATH_TO_ALL_BENCH_CSV)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please download it from the link in the README.")
    
    logger.debug(f"Found {results_csv}")


if __name__ == "__main__":
    main()