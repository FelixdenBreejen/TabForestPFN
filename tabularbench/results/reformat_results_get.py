from pathlib import Path

import xarray as xr

from tabularbench.core.enums import BenchmarkOrigin
from tabularbench.utils.paths_and_filenames import (PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED,
                                                    PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)


def get_reformatted_results(benchmark_origin: BenchmarkOrigin) -> xr.Dataset:

    match benchmark_origin:
        case BenchmarkOrigin.TABZILLA:
            return get_reformatted_results_tabzilla()
        case BenchmarkOrigin.WHYTREES:
            return get_reformatted_results_whytrees()
        case _:
            raise ValueError(f"Unknown benchmark origin {benchmark_origin}")


def get_reformatted_results_tabzilla() -> xr.Dataset:

    if not Path(PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED).exists():
        raise FileNotFoundError(f"File {PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED} does not exist, did you run reformat_results_whytrees()?")

    return xr.open_dataset(PATH_TO_TABZILLA_BENCH_RESULTS_REFORMATTED)


def get_reformatted_results_whytrees() -> xr.Dataset:

    if not Path(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED).exists():
        raise FileNotFoundError(f"File {PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED} does not exist, did you run reformat_results_whytrees()?")

    return xr.open_dataset(PATH_TO_WHYTREES_BENCH_RESULTS_REFORMATTED)