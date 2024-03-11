import functools

from loguru import logger

from tabularbench.core.enums import DataSplit, Task
from tabularbench.results.reformat_whytrees_benchmark import get_whytrees_benchmark_reformatted
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
    score_min, score_max = scores_min_max_(openml_dataset_id, benchmark_model_names, cfg.benchmark.task, data_split)

    return score_min, score_max



@functools.lru_cache(maxsize=None)
def scores_min_max_(openml_dataset_id: int, benchmark_model_names: tuple[str], task: Task, data_split: DataSplit) -> tuple[float, float]:

    ds_whytrees = get_whytrees_benchmark_reformatted()
    ds_whytrees = ds_whytrees.sel(model_name=list(benchmark_model_names), openml_dataset_id=openml_dataset_id, data_split=data_split.name)

    match task:
        case Task.REGRESSION:
            score_min = ds_whytrees['score'].quantile(0.5).values.item()
        case Task.CLASSIFICATION:
            score_min = ds_whytrees['score'].quantile(0.1).values.item()

    score_max = ds_whytrees['score'].max().values.item()

    logger.info(f"For dataset id {openml_dataset_id} and split {data_split}, we will normalize with min {score_min:.4f} and max {score_max:.4f}")

    return score_min, score_max
