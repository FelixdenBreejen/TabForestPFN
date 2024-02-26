from __future__ import annotations
from dataclasses import dataclass
import dataclasses
from pathlib import Path
from typing import Self
from omegaconf import DictConfig, OmegaConf

import torch
import yaml

from tabularbench.core.enums import ModelName, Task, DatasetSize
from tabularbench.data.datafile_openml import OpenmlDatafile
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


@dataclass
class ConfigRun():
    output_dir: Path
    device: torch.device
    seed: int
    model_name: ModelName
    task: Task
    dataset_size: DatasetSize
    openml_dataset_id: int
    openml_dataset_name: str
    datafile_path: Path
    hyperparams: DictConfig


    @classmethod
    def create(
            cls, 
            cfg: ConfigBenchmarkSweep, 
            seed: int,
            device: torch.device, 
            dataset_file_path: Path,
            hyperparams: DictConfig,
            run_id: int
        ) -> Self:

        dataset_size = cfg.benchmark.dataset_size
        openml_datafile = OpenmlDatafile(dataset_file_path)
        openml_dataset_id = openml_datafile.ds.attrs['openml_dataset_id']
        openml_dataset_name = openml_datafile.ds.attrs['openml_dataset_name']
        
        output_dir = cfg.output_dir / str(openml_dataset_id) / f"#{run_id}"

        return cls(
            output_dir=output_dir,
            model_name=cfg.model_name,
            device=device,
            seed=seed,
            task=cfg.benchmark.task,
            dataset_size=dataset_size,
            openml_dataset_id=openml_dataset_id,
            openml_dataset_name=openml_dataset_name,
            datafile_path=dataset_file_path,
            hyperparams=hyperparams
        )
    
    def save(self) -> None:
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cfg = dataclasses.replace(self, hyperparams=OmegaConf.to_container(self.hyperparams, resolve=True))

        with open(self.output_dir / "config_run.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

            