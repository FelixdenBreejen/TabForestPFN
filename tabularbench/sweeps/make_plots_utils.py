from __future__ import annotations

import pandas as pd

from tabularbench.core.enums import SearchType
from tabularbench.utils.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.utils.paths_and_filenames import \
    DEFAULT_RESULTS_TEST_FILE_NAME


def sweep_default_finished(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    df = df_run_results
    df = df[ df['search_type'] == SearchType.DEFAULT.name ]
    df = df[ df['seed'] == cfg.seed ]    # when using multiple default runs, the seed changes

    for dataset_id in cfg.openml_dataset_ids_to_use:

        df_id = df[ df['openml_dataset_id'] == dataset_id ]
        if len(df_id) == 0:
            return False

    return True


def default_results_not_yet_made(cfg: ConfigBenchmarkSweep) -> bool:
    return not (cfg.output_dir / DEFAULT_RESULTS_TEST_FILE_NAME).exists()


