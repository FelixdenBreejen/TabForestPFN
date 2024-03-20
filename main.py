from __future__ import annotations

from pathlib import Path

import hydra
import torch.multiprocessing as mp
from loguru import logger
from omegaconf import DictConfig

from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.utils.config_main import ConfigMain
from tabularbench.utils.paths_and_filenames import (CONFIG_MAIN_FILE_NAME, PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED,
                                                    PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)
from tabularbench.utils.set_seed import set_seed


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_hydra: DictConfig):
    
    mp.set_start_method('spawn')

    cfg = ConfigMain.from_hydra(cfg_hydra)

    logger.add(cfg.output_dir / "log.log", enqueue=True)
    logger.info("Finished creating main config")

    check_existence_of_benchmark_results_csv(cfg)
    cfg.save(path=cfg.output_dir / CONFIG_MAIN_FILE_NAME)
    set_seed(cfg.seed)


    for cfg_benchmark_sweep in cfg.configs_benchmark_sweep:

        logger.info(f"Start benchmark sweep for {cfg_benchmark_sweep.benchmark.name}")

        run_sweep(cfg_benchmark_sweep)

        logger.info(f"Finished benchmark sweep for {cfg_benchmark_sweep.benchmark.name}")
    
    logger.info("Finished all sweeps")



def check_existence_of_benchmark_results_csv(cfg: ConfigMain) -> None:

    results_csv = Path(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please preprocess the data using the preprocess.py file.")
    
    logger.debug(f"Found {results_csv}")

    results_csv = Path(PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please preprocess the data using the preprocess.py file.")
        
    logger.debug(f"Found {results_csv}")
    


if __name__ == "__main__":
    main()