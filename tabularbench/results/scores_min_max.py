import functools
import logging

from tabularbench.core.enums import DataSplit, Task
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep


def get_combined_normalized_scores(cfg: ConfigBenchmarkSweep, openml_ids: list[int], data_split: DataSplit, scores: list[float]) -> float:
    """
    Based on a list of scores belonging to dataset ids, we compute the normalized test score.
    """

    normalized_scores = []

    for openml_dataset_id, score in zip(openml_ids, scores):
        score_min, score_max = scores_min_max(cfg, openml_dataset_id, data_split)
        normalized_score = (score - score_min) / (score_max - score_min)
        normalized_score = max(0.0, normalized_score)
        normalized_scores.append(normalized_score)

    combined_normalized_score = sum(normalized_scores) / len(normalized_scores)
    return combined_normalized_score


def scores_min_max(cfg: ConfigBenchmarkSweep, openml_dataset_id: int, data_split: DataSplit) -> tuple[float, float]:
    """
    Based on the benchmark results, we normalize the scores of the sweep.
    Returns the min and max scores to normalize with
    """

    benchmark_model_names = tuple(model_name.name for model_name in cfg.config_plotting.benchmark_model_names)
    score_min, score_max = scores_min_max_(openml_dataset_id, benchmark_model_names, cfg.benchmark.task, data_split, logger)

    return score_min, score_max



@functools.lru_cache(maxsize=None)
def scores_min_max_(openml_dataset_id: int, benchmark_model_names: tuple[str], task: Task, data_split: DataSplit, logger: logging.Logger) -> tuple[float, float]:

    match data_split:
        case DataSplit.VALID:
            score_name = 'score_val_mean'
        case DataSplit.TEST:
            score_name = 'score_test_mean'

    df_bench = get_benchmark_csv_reformatted()
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench = df_bench[ df_bench['openml_dataset_id'] == openml_dataset_id ]

    match task:
        case Task.REGRESSION:
            score_min = df_bench[score_name].quantile(0.50)
        case Task.CLASSIFICATION:
            score_min = df_bench[score_name].quantile(0.10)

    score_max = df_bench[score_name].max()

    logger.info(f"For dataset id {openml_dataset_id} and split {data_split}, we will normalize with min {score_min:.4f} and max {score_max:.4f}")

    return score_min, score_max
