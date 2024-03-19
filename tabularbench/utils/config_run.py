from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from tabularbench.core.enums import DatasetSize, ModelName, Task
from tabularbench.data.datafile_openml import OpenmlDatafile
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.config_save_load_mixin import ConfigSaveLoadMixin


@dataclass
class ConfigRun(ConfigSaveLoadMixin):
    output_dir: Path
    device: torch.device
    seed: int
    model_name: ModelName
    task: Task
    dataset_size: DatasetSize
    openml_dataset_id: int
    openml_dataset_name: str
    datafile_path: Path
    hyperparams: dict


    @classmethod
    def create(
        cls, 
        cfg: ConfigBenchmarkSweep, 
        seed: int,
        device: torch.device, 
        dataset_file_path: Path,
        hyperparams: dict,
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

            