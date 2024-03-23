from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from tabularbench.config.config_benchmark_sweep import ConfigPlotting
from tabularbench.config.config_data import ConfigData
from tabularbench.config.config_optim import ConfigOptim
from tabularbench.config.config_plotting import ConfigPlottingTabzilla, ConfigPlottingWhytrees
from tabularbench.config.config_preprocessing import ConfigPreprocessing
from tabularbench.config.config_save_load_mixin import ConfigSaveLoadMixin
from tabularbench.config.config_testing import ConfigTesting
from tabularbench.core.enums import BenchmarkName, GeneratorName, ModelName


@dataclass
class ConfigPretrain(ConfigSaveLoadMixin):
    output_dir: Path
    seed: int
    devices: list[torch.device]
    use_ddp: bool
    workers_per_gpu: int
    model: dict
    model_name: ModelName
    data: ConfigData
    optim: ConfigOptim
    preprocessing: ConfigPreprocessing
    testing: ConfigTesting
    plotting: ConfigPlotting
    hyperparams_finetuning: dict

    device: Optional[torch.device] = None   # initialized later by ddp
    is_main_process: bool = True            # initialized later by ddp


    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):

        output_dir = Path(cfg_hydra.output_dir)

        devices = [torch.device(device) for device in cfg_hydra.devices]
        pretrain_model_name = ModelName[cfg_hydra.pretrain_model.name]
        hyperparams_finetuning = cfg_hydra.hyperparams[pretrain_model_name.name.lower()]
        model_settings = cfg_hydra.pretrain_model


        return cls(
            output_dir=output_dir,
            devices=devices,
            use_ddp=len(devices) > 1,
            seed=cfg_hydra.seed,
            workers_per_gpu=cfg_hydra.workers_per_gpu,
            model = OmegaConf.to_container(model_settings),
            model_name = pretrain_model_name,
            hyperparams_finetuning = OmegaConf.to_container(hyperparams_finetuning),
            data = ConfigData(
                generator=GeneratorName(cfg_hydra.data.generator),
                min_samples_support=cfg_hydra.data.min_samples_support,
                max_samples_support=cfg_hydra.data.max_samples_support,
                n_samples_query=cfg_hydra.data.n_samples_query,
                min_features=cfg_hydra.data.min_features,
                max_features=cfg_hydra.data.max_features,
                max_classes=cfg_hydra.data.max_classes,
                generator_hyperparams=OmegaConf.to_container(cfg_hydra.data.generator_hyperparams),
            ),
            optim = ConfigOptim(
                max_steps=cfg_hydra.optim.max_steps,
                log_every_n_steps=cfg_hydra.optim.log_every_n_steps,
                eval_every_n_steps=cfg_hydra.optim.eval_every_n_steps,
                batch_size=cfg_hydra.optim.batch_size,
                gradient_accumulation_steps=cfg_hydra.optim.gradient_accumulation_steps,
                lr=cfg_hydra.optim.lr,
                weight_decay=cfg_hydra.optim.weight_decay,
                beta1=cfg_hydra.optim.beta1,
                beta2=cfg_hydra.optim.beta2,
                warmup_steps=cfg_hydra.optim.warmup_steps,
                cosine_scheduler=cfg_hydra.optim.cosine_scheduler,
                max_grad_norm=cfg_hydra.optim.max_grad_norm,
                use_pretrained_weights=cfg_hydra.optim.use_pretrained_weights,
                path_to_weights=cfg_hydra.optim.path_to_weights,
            ),
            preprocessing = ConfigPreprocessing(
                use_quantile_transformer=cfg_hydra.preprocessing.use_quantile_transformer,
                use_feature_count_scaling=cfg_hydra.preprocessing.use_feature_count_scaling,
            ),
            testing = ConfigTesting(
                n_default_runs_per_dataset_valid=cfg_hydra.testing.n_default_runs_per_dataset_valid,
                n_default_runs_per_dataset_test=cfg_hydra.testing.n_default_runs_per_dataset_test,
                openml_dataset_ids_to_ignore=OmegaConf.to_container(cfg_hydra.testing.openml_dataset_ids_to_ignore),
                benchmarks=[BenchmarkName[benchmark] for benchmark in cfg_hydra.testing.benchmarks],
            ),
            plotting = ConfigPlotting(
                whytrees = ConfigPlottingWhytrees(                    
                    n_runs=cfg_hydra.plotting.whytrees.n_runs,
                    n_random_shuffles=cfg_hydra.plotting.whytrees.n_random_shuffles,
                    confidence_bound=cfg_hydra.plotting.whytrees.confidence_bound,
                    plot_default_value=cfg_hydra.plotting.whytrees.plot_default_value,
                    benchmark_model_names=[ModelName[model] for model in cfg_hydra.plotting.whytrees.benchmark_models]
                ),
                tabzilla = ConfigPlottingTabzilla(
                    benchmark_model_names=[ModelName[model] for model in cfg_hydra.plotting.tabzilla.benchmark_models],
                )
            ),
        )
    











