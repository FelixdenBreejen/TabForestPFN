import functools
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from tabularbench.results.run_results import RunResults

from tabularbench.utils.paths_and_filenames import PATH_TO_ALL_BENCH_CSV, PATH_TO_ALL_BENCH_CSV_REFORMATTED


def reformat_benchmark():

    path = Path(PATH_TO_ALL_BENCH_CSV)

    assert path.exists(), f"File {path} does not exist, did you download the benchmark?"
    
    df = pd.read_csv(path)
    rows_new = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        results = RunResults.from_benchmark_row(row)

        if results is None:
            continue

        rows_new.append(results.to_dict())
    
    df_new = pd.DataFrame(rows_new)
    df_new.to_csv(PATH_TO_ALL_BENCH_CSV_REFORMATTED, index=False)


@functools.lru_cache(maxsize=1)
def get_benchmark_csv_reformatted():

    if not Path(PATH_TO_ALL_BENCH_CSV_REFORMATTED).exists():
        reformat_benchmark()

    df = pd.read_csv(PATH_TO_ALL_BENCH_CSV_REFORMATTED, low_memory=False)
    return df
    


if __name__ == "__main__":
    reformat_benchmark()