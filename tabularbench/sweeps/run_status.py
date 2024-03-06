
from tabularbench.results.results_run import ResultsRun
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def no_runs_finished(run_results_dict: dict[int, list[ResultsRun]]) -> bool:
    return all([len(run_results_dict[dataset_id]) == 0 for dataset_id in run_results_dict])


def all_runs_finished(cfg: ConfigBenchmarkSweep, run_results_dict: dict[int, list[ResultsRun]], runs_per_dataset: int) -> bool:
    # All runs are finished if there are no datasets left with less than runs_per_dataset runs

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) < runs_per_dataset and dataset_id not in cfg.openml_dataset_ids_to_ignore:
            return False
    else:
        return True


def all_runs_almost_finished(cfg: ConfigBenchmarkSweep, run_results_dict: dict[int, list[ResultsRun]], runs_per_dataset: int, runs_busy_dict: dict[int, int]) -> bool:
    # All runs are almost finished if there are no datasets left if all runs that are currently running are finished

    for dataset_id in run_results_dict.keys():
        if len(run_results_dict[dataset_id]) + runs_busy_dict[dataset_id] < runs_per_dataset and dataset_id not in cfg.openml_dataset_ids_to_ignore:
            return False
    else:
        return True