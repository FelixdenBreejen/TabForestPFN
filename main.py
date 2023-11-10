from __future__ import annotations

import time
import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch.multiprocessing as mp
import yaml
from tabularbench.sweeps.config_main import ConfigMain

from tabularbench.sweeps.monitor_and_make_plots import monitor_and_make_plots
from tabularbench.sweeps.run_sweeps import run_sweeps
from tabularbench.sweeps.paths_and_filenames import PATH_TO_ALL_BENCH_CSV, CONFIG_DUPLICATE
from tabularbench.sweeps.writer import file_writer



@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_hydra: DictConfig):

    cfg = ConfigMain.from_hydra(cfg_hydra)
    cfg.logger.info("Finished creating main config")

    check_existence_of_benchmark_results_csv(cfg)
    save_config(cfg)


    launch_sweeps(cfg)


def save_config(cfg: ConfigMain) -> None:
    
    config_path = Path(cfg.output_dir) / CONFIG_DUPLICATE

    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


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


def check_existence_of_benchmark_results_csv(cfg: ConfigMain) -> None:

    results_csv = Path(PATH_TO_ALL_BENCH_CSV)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please download it from the link in the README.")
    
    cfg.logger.debug(f"Found {results_csv}")


if __name__ == "__main__":
    main()