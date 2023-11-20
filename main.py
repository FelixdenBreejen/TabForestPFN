from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch.multiprocessing as mp
import yaml
from tabularbench.sweeps.config_main import ConfigMain

from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.sweeps.paths_and_filenames import PATH_TO_ALL_BENCH_CSV, CONFIG_DUPLICATE
from tabularbench.sweeps.sweep_start import set_seed



@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg_hydra: DictConfig):
    
    mp.set_start_method('spawn')

    cfg = ConfigMain.from_hydra(cfg_hydra)
    cfg.logger.info("Finished creating main config")

    check_existence_of_benchmark_results_csv(cfg)
    save_config(cfg)
    set_seed(cfg.seed)
    # TODO: check how seeding works in subprocesses


    for cfg_benchmark_sweep in cfg.configs_benchmark_sweep:

        cfg.logger.info(f"Start benchmark sweep for {cfg_benchmark_sweep.benchmark.name}")

        run_sweep(cfg_benchmark_sweep)

        cfg.logger.info(f"Finished benchmark sweep for {cfg_benchmark_sweep.benchmark.name}")
    
    cfg.logger.info("Finished all sweeps")




def save_config(cfg: ConfigMain) -> None:
    
    config_path = Path(cfg.output_dir) / CONFIG_DUPLICATE

    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


def check_existence_of_benchmark_results_csv(cfg: ConfigMain) -> None:

    results_csv = Path(PATH_TO_ALL_BENCH_CSV)
    if not results_csv.exists():
        raise FileNotFoundError(f"Could not find {results_csv}. Please download it from the link in the README.")
    
    cfg.logger.debug(f"Found {results_csv}")


if __name__ == "__main__":
    main()