from dataclasses import dataclass
from typing import Self

from tabularbench.core.enums import BenchmarkName, DatasetSize, Task
from tabularbench.data.datafile_openml import OpenmlDatafile


@dataclass
class Benchmark:
    name: BenchmarkName
    task: Task
    dataset_size: DatasetSize
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]

    @classmethod
    def create(cls, name: BenchmarkName, task: Task, dataset_size: DatasetSize, openml_dataset_ids: list[int]) -> Self:

        dataset_names = []
        for openml_dataset_id in openml_dataset_ids:
            dataset_name = OpenmlDatafile(openml_dataset_id, dataset_size).ds.attrs['openml_dataset_name']
            dataset_names.append(dataset_name)

        return cls(
            name=name,
            task=task,
            dataset_size=dataset_size,
            openml_dataset_ids=openml_dataset_ids,
            openml_dataset_names=dataset_names
        )




BENCHMARKS = {
    BenchmarkName.CATEGORICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44156, 44157, 44159, 45035, 45036, 45038, 45039]
    ),
    BenchmarkName.CATEGORICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44055, 44056, 44059, 44061, 44062, 44063, 44065, 44066, 44068, 44069, 45041, 45042, 45043, 45045, 45046, 45047, 45048]
    ),
    BenchmarkName.NUMERICAL_REGRESSION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44132, 44133, 44134, 44136, 44137, 44138, 44139, 44140, 44141, 44142, 44143, 44144, 44145, 44146, 44147, 44148, 45032, 45033, 45034]
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.MEDIUM,
        openml_dataset_ids=[44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022, 45021, 45020, 45019, 45028, 45026]
    ),
    BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_CLASSIFICATION_LARGE,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44156, 44157, 44159, 45035, 45036, 45038, 45039]
    ),
    BenchmarkName.CATEGORICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.CATEGORICAL_REGRESSION_LARGE,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44055, 44056, 44059, 44061, 44062, 44063, 44065, 44066, 44068, 44069, 45041, 45042, 45043, 45045, 45046, 45047, 45048]
    ),
    BenchmarkName.NUMERICAL_REGRESSION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_REGRESSION_LARGE,
        task=Task.REGRESSION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44132, 44133, 44134, 44136, 44137, 44138, 44139, 44140, 44141, 44142, 44143, 44144, 44145, 44146, 44147, 44148, 45032, 45033, 45034]
    ),
    BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE: Benchmark.create(
        name=BenchmarkName.NUMERICAL_CLASSIFICATION_LARGE,
        task=Task.CLASSIFICATION,
        dataset_size=DatasetSize.LARGE,
        openml_dataset_ids=[44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022, 45021, 45020, 45019, 45028, 45026]
    ),
}


if __name__ == "__main__":
    print(BENCHMARKS)
    pass