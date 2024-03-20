from pathlib import Path

from loguru import logger

from tabularbench.data.tabzilla_data_preprocessing import preprocess_tabzilla_data
from tabularbench.data.whytrees_data_preprocessing import preprocess_whytrees_data
from tabularbench.results.reformat_results_tabzilla import reformat_results_tabzilla
from tabularbench.results.reformat_results_whytrees import reformat_results_whytrees
from tabularbench.utils.paths_and_filenames import PATH_TO_OPENML_DATASETS


def main():

    data_path = Path(PATH_TO_OPENML_DATASETS)

    if not data_path.exists():
        data_path.mkdir(parents=True)

    logger.info(f"Datasets will be saved to {data_path}.")

    logger.info("Preprocessing whytrees datasets...")
    preprocess_whytrees_data()
    logger.info("Preprocessing whytrees datasets... done.")
    logger.info("Preprocessing tabzilla datasets...")
    preprocess_tabzilla_data()
    logger.info("Preprocessing tabzilla datasets... done.")
    logger.info("Reformat whytrees benchmark...")
    reformat_results_whytrees()
    logger.info("Reformat whytrees benchmark... done.")
    logger.info("Reformat tabzilla benchmark...")
    reformat_results_tabzilla()
    logger.info("Reformat tabzilla benchmark... done.")


if __name__ == "__main__":
    main()