from pathlib import Path

# Import all openml preprocessor modules.
# NOTE: To import datasets from sources other than openml, add them using a new module
from tabularbench.data.tabzilla_preprocessors_openml import preprocessor_dict

import warnings
warnings.filterwarnings('ignore', category=FutureWarning ) # Openml FutureWarning for version 0.15

dataset_path = Path("data/openml/")


def build_preprocessors_dict():
    preprocessors = {}
    duplicates = preprocessors.keys() & preprocessor_dict.keys()
    if duplicates:
        raise RuntimeError(
            f"Duplicate dataset_name key found preprocessor dict: {duplicates}"
        )
    preprocessors.update(preprocessor_dict)
    return preprocessors


preprocessors = build_preprocessors_dict()


def preprocess_dataset(dataset_name, overwrite=False, verbose=True):
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

    process_all = True


    print("------------------------\n")
    print("Valid dataset names:\n")
    for i, dataset_name in enumerate(sorted(preprocessors.keys())):
        print(f"{i + 1}: {dataset_name} ")
    print("------------------------")

    for dataset_name in sorted(preprocessors.keys()):
        _ = preprocess_dataset(dataset_name, overwrite=True)
        print("Processed dataset {}".format(dataset_name))
