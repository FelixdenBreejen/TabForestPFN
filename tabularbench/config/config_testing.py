from dataclasses import dataclass

from tabularbench.core.enums import BenchmarkName


@dataclass
class ConfigTesting():
    n_default_runs_per_dataset_valid: int
    n_default_runs_per_dataset_test: int               
    openml_dataset_ids_to_ignore: list[int]
    benchmarks: list[BenchmarkName]