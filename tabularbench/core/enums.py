from enum import IntEnum, StrEnum


class Task(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class FeatureType(StrEnum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    MIXED = "mixed"


class SearchType(StrEnum):
    DEFAULT = "default"
    RANDOM = "random"


class DatasetSize(IntEnum):
    SMALL = 1000
    MEDIUM = 10000
    LARGE = 50000


class ModelName(StrEnum):
    FT_TRANSFORMER = "FT_Transformer"
    TABPFN_FINETUNE = "TabPFN_Finetune"
    SAINT = "SAINT"
    MLP = "MLP"
    RESNET = "Resnet"
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    LIGHTGBM = "lightGBM"
    GRADIENT_BOOSTING_TREE = "GradientBoostingTree"
    HIST_GRADIENT_BOOSTING_TREE = "HistGradientBoostingTree"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"


class BenchmarkName(StrEnum):
    CATEGORICAL_CLASSIFICATION = "categorical_classification"
    NUMERICAL_CLASSIFICATION = "numerical_classification"
    CATEGORICAL_REGRESSION = "categorical_regression"
    NUMERICAL_REGRESSION = "numerical_regression"
    CATEGORICAL_CLASSIFICATION_LARGE = "categorical_classification_large"
    NUMERICAL_CLASSIFICATION_LARGE = "numerical_classification_large"
    CATEGORICAL_REGRESSION_LARGE = "categorical_regression_large"
    NUMERICAL_REGRESSION_LARGE = "numerical_regression_large"



