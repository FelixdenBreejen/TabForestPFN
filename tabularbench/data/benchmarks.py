from dataclasses import dataclass
from typing import Self

import openml
from tabularbench.core.enums import BenchmarkName, DatasetSize, FeatureType, Task


@dataclass
class Benchmark:
    name: BenchmarkName
    task: Task
    dataset_size: DatasetSize
    categorical: FeatureType
    openml_suite_id: int
    openml_task_ids: list[int]
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]

    @classmethod
    def create(cls, name: BenchmarkName, task: Task, dataset_size: DatasetSize, categorical: FeatureType, suite_id: int) -> Self:

        openml_suite = openml.study.get_suite(suite_id)
        openml_task_ids = openml_suite.tasks
        assert openml_task_ids is not None

        openml_dataset_ids = openml_suite.data
        assert openml_dataset_ids is not None

        openml_dataset_names = []
        
        for dataset_id in openml_dataset_ids:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)
            openml_dataset_names.append(dataset.name)

        return cls(
            name=name,
            task=task,
            dataset_size=dataset_size,
            categorical=categorical,
            openml_suite_id=suite_id,
            openml_task_ids=openml_task_ids,
            openml_dataset_ids=openml_dataset_ids,
            openml_dataset_names=openml_dataset_names
        )




BENCHMARKS = {
    BenchmarkName.CATEGORICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        categorical=FeatureType.MIXED,
        suite_id=334
    ),
    BenchmarkName.CATEGORICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        categorical=FeatureType.MIXED,
        suite_id=335
    ),
    BenchmarkName.NUMERICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        categorical=FeatureType.NUMERICAL,
        suite_id=336
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        categorical=FeatureType.NUMERICAL,
        suite_id=337
    ),
    BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        categorical=FeatureType.MIXED,
        suite_id=334
    ),
    BenchmarkName.CATEGORICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION_LARGE,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        categorical=FeatureType.MIXED,
        suite_id=335
    ),
    BenchmarkName.NUMERICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION_LARGE,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        categorical=FeatureType.NUMERICAL,
        suite_id=336
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        categorical=FeatureType.NUMERICAL,
        suite_id=337
    ),
}