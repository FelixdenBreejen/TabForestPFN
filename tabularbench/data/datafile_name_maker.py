
from pathlib import Path
from typing import Optional
from tabularbench.core.enums import BenchmarkOrigin, DatasetSize
from tabularbench.utils.paths_and_filenames import PATH_TO_OPENML_DATASETS


def make_datafile_path(origin: BenchmarkOrigin, openml_dataset_id: int, dataset_size: Optional[DatasetSize]) -> str:

    validate_arguments(origin, dataset_size)
    datafile_name = make_datafile_name(origin, openml_dataset_id, dataset_size)
    datafile_path = Path(PATH_TO_OPENML_DATASETS) / datafile_name
    return datafile_path


def validate_arguments(origin: BenchmarkOrigin, dataset_size: Optional[DatasetSize]) -> None:

    match origin:
        case BenchmarkOrigin.WHYTREES:
            # Only the why trees benchmark has different dataset sizes.
            assert dataset_size is not None
        case BenchmarkOrigin.TABZILLA:
            assert dataset_size is None


def make_datafile_name(origin: BenchmarkOrigin, openml_dataset_id: int, dataset_size: Optional[DatasetSize]) -> str:

    match origin:
        case BenchmarkOrigin.WHYTREES:
            return f"whytrees_{openml_dataset_id}_{dataset_size.name}.nc"
        case BenchmarkOrigin.TABZILLA:
            return f"tabzilla_{openml_dataset_id}.nc"
