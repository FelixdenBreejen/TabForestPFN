import functools
import logging

from tabularbench.core.enums import Task
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep


def scores_min_max(cfg: ConfigBenchmarkSweep, openml_dataset_id: int) -> tuple[float, float]:
    """
    Based on the benchmark results, we normalize the scores of the sweep.
    Returns the min and max scores to normalize with
    """

    benchmark_model_names = tuple(model_name.name for model_name in cfg.config_plotting.benchmark_model_names)
    score_min, score_max = scores_min_max_(openml_dataset_id, benchmark_model_names, cfg.benchmark.task, cfg.logger)

    return score_min, score_max



@functools.lru_cache(maxsize=None)
def scores_min_max_(openml_dataset_id: int, benchmark_model_names: tuple[str], task: Task, logger: logging.Logger) -> tuple[float, float]:

    df_bench = get_benchmark_csv_reformatted()
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench = df_bench[ df_bench['openml_dataset_id'] == openml_dataset_id ]

    match task:
        case Task.REGRESSION:
            score_min = df_bench['score_test_mean'].quantile(0.50)
        case Task.CLASSIFICATION:
            score_min = df_bench['score_test_mean'].quantile(0.10)

    score_max = df_bench['score_test_mean'].max()

    logger.info(f"For dataset id {openml_dataset_id}, we will normalize with min {score_min:.4f} and max {score_max:.4f}")

    return score_min, score_max
