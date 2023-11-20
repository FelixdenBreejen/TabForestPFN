from __future__ import annotations
from dataclasses import dataclass
import itertools
import logging
from pathlib import Path

from omegaconf import DictConfig
import torch


from tabularbench.core.enums import BenchmarkName, ModelName, SearchType
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep, ConfigPlotting
from tabularbench.sweeps.get_logger import get_logger


@dataclass
class ConfigMain():
    logger: logging.Logger
    output_dir: Path
    seed: int
    configs_benchmark_sweep: list[ConfigBenchmarkSweep]


    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        output_dir = Path(cfg_hydra.output_dir)
        logger = get_logger(output_dir / 'log.txt')
        logger.info(f"Start creating main config")
        configs_benchmark_sweep = cls.create_configs_benchmark_sweep(cfg_hydra, output_dir, logger)
        logger.info(f"Finished creating main config")

        return cls(
            logger=logger,
            output_dir=output_dir,
            seed=cfg_hydra.seed,
            configs_benchmark_sweep=configs_benchmark_sweep
        )
    
    
    @staticmethod
    def create_configs_benchmark_sweep(cfg_hydra: DictConfig, output_dir: Path, logger: logging.Logger) -> list[ConfigBenchmarkSweep]:

        benchmark_names = [BenchmarkName[benchmark] for benchmark in cfg_hydra.benchmarks]
        models = [ModelName[model] for model in cfg_hydra.models]
        search_types = [SearchType[search_type] for search_type in cfg_hydra.search_types]

        devices = [torch.device(device) for device in cfg_hydra.devices]

        assert len(models) == len(cfg_hydra.model_plot_names), f"Please provide a plot name for each model. Got {len(models)} models and {len(cfg_hydra.model_plot_names)} plot names."
        models_with_plot_name = zip(models, cfg_hydra.model_plot_names)
        sweep_details = itertools.product(models_with_plot_name, search_types, benchmark_names)

        config_plotting = ConfigPlotting(
            n_runs=cfg_hydra.plotting.n_runs,
            n_random_shuffles=cfg_hydra.plotting.n_random_shuffles,
            confidence_bound=cfg_hydra.plotting.confidence_bound,
            benchmark_models=[ModelName[model] for model in cfg_hydra.plotting.benchmark_models],
        )

        benchmark_sweep_configs = []

        for (model_name, model_plot_name), search_type, benchmark_name in sweep_details:

            benchmark = BENCHMARKS[benchmark_name]
            hyperparams_object = cfg_hydra.hyperparams[model_name.name.lower()]

            output_dir_benchmark = output_dir / f'{model_name.value.lower()}-{search_type.value}-{benchmark_name.value}'
            logger_benchmark = get_logger(output_dir_benchmark / 'log.txt')

            dataset_ids_to_ignore = list(set(cfg_hydra.openml_dataset_ids_to_ignore) & set(benchmark.openml_dataset_ids))

            bscfg = ConfigBenchmarkSweep(
                logger=logger_benchmark,
                output_dir=output_dir_benchmark,
                seed=cfg_hydra.seed,
                devices=devices,
                benchmark=benchmark,
                model_name=model_name,
                model_plot_name=model_plot_name,
                search_type=search_type,
                config_plotting=config_plotting,
                n_random_runs_per_dataset=cfg_hydra.n_random_runs_per_dataset,
                n_default_runs_per_dataset=cfg_hydra.n_default_runs_per_dataset,
                openml_dataset_ids_to_ignore=dataset_ids_to_ignore,
                hyperparams_object=hyperparams_object
            )
        
            logger.info(f"Created benchmark sweep config for {bscfg.benchmark.name}-{bscfg.model_name.name}-{bscfg.search_type.name} ")
            benchmark_sweep_configs.append(bscfg)
        
        return benchmark_sweep_configs




