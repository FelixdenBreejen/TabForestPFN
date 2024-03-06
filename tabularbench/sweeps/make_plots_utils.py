from __future__ import annotations

from tabularbench.results.results_sweep import ResultsSweep
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import \
    DEFAULT_RESULTS_TEST_FILE_NAME


def sweep_default_finished(cfg: ConfigBenchmarkSweep, results_sweep: ResultsSweep) -> None:

    for dataset_id in cfg.openml_dataset_ids_to_use:
        if not dataset_id in results_sweep.ds.coords['openml_dataset_id']:
            return False
        
    all_datasets_at_least_one_run = results_sweep.ds['runs_actual'].all() 
        
    return all_datasets_at_least_one_run


def default_results_not_yet_made(cfg: ConfigBenchmarkSweep) -> bool:
    return not (cfg.output_dir / DEFAULT_RESULTS_TEST_FILE_NAME).exists()


