benchmarks = [
    {
        "task": "regression",
        "dataset_size": "medium",
        "categorical": False,
        "name": "numerical_regression",
        "suite_id": 336,
        "exclude": []
    },
    {
        "task": "regression",
        "dataset_size": "large",
        "categorical": False,
        "name": "numerical_regression_large",
        "suite_id": 336,
        "exclude": []
    },
    {
        "task": "classif",
        "dataset_size": "medium",
        "categorical": False,
        "name": "numerical_classification",
        "suite_id": 337,
        "exlude": []
    },
    {
        "task": "classif",
        "dataset_size": "large",
        "categorical": False,
        "name": "numerical_classification_large",
        "suite_id": 337,
        "exclude": []
    },
    {
        "task": "regression",
        "dataset_size": "medium",
        "categorical": True,
        "name": "categorical_regression",
        "suite_id": 335,
        "exclude": [],
    },
    {
        "task": "regression",
        "dataset_size": "large",
        "categorical": True,
        "name": "categorical_regression_large",
        "suite_id": 335,
        "exclude": [],
    },
    {
        "task": "classif",
        "dataset_size": "medium",
        "categorical": True,
        "name": "categorical_classification",
        "suite_id": 334,
        "exclude": [],
    },
    {
        "task": "classif",
        "dataset_size": "large",
        "categorical": True,
        "name": "categorical_classification_large",
        "suite_id": 334,
        "exclude": [],
    }
]


benchmark_names = [benchmark['name'] for benchmark in benchmarks]