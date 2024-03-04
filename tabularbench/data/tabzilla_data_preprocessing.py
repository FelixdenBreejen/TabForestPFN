from pathlib import Path
from loguru import logger

# Import all openml preprocessor modules.
# NOTE: To import datasets from sources other than openml, add them using a new module
from tabularbench.data.tabzilla_preprocessors_openml import create_preprocessor_dict

import warnings

from tabularbench.utils.paths_and_filenames import PATH_TO_OPENML_DATASETS
warnings.filterwarnings('ignore', category=FutureWarning ) # Openml FutureWarning for version 0.15

dataset_path = Path(PATH_TO_OPENML_DATASETS)


def preprocess_tabzilla_data():

    preprocessors = build_preprocessors_dict()

    logger.info("------------------------\n")
    logger.info("Valid dataset names:\n")
    for i, dataset_name in enumerate(sorted(preprocessors.keys())):
        logger.info(f"{i + 1}: {dataset_name} ")
    logger.info("------------------------")

    for dataset_name in sorted(preprocessors.keys()):
        _ = preprocess_dataset(dataset_name, preprocessors, overwrite=True)
        logger.info("Processed dataset {}".format(dataset_name))




def build_preprocessors_dict():

    preprocessor_dict = create_preprocessor_dict()

    preprocessors = {}
    duplicates = preprocessors.keys() & preprocessor_dict.keys()
    if duplicates:
        raise RuntimeError(
            f"Duplicate dataset_name key found preprocessor dict: {duplicates}"
        )
    preprocessors.update(preprocessor_dict)
    return preprocessors


def preprocess_dataset(dataset_name, preprocessors, overwrite=False, verbose=True):
    dest_path = dataset_path / dataset_name
    if not overwrite and dest_path.exists():
        if verbose:
            print(f"{dataset_name:<40}| Found existing folder. Skipping.")
        return dest_path

    print(f"{dataset_name:<40}| Processing...")
    if dataset_name not in preprocessors:
        raise KeyError(f"Unrecognized dataset name: {dataset_name}")
    dataset = preprocessors[dataset_name]()
    dataset.write(dataset_path, overwrite=overwrite)
    return dataset_path


if __name__ == "__main__":
    preprocess_tabzilla_data()
