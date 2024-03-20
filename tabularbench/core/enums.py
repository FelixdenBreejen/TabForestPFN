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


class DataSplit(StrEnum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class ModelName(StrEnum):
    PLACEHOLDER = "_placeholder_"   # This is a placeholder for the current running model
    FT_TRANSFORMER = "FT_Transformer"
    TABPFN = "TabPFN"
    FOUNDATION = "Foundation"
    SAINT = "SAINT"
    MLP = "MLP"
    MLP_RTDL = "MLP-rtdl"
    RESNET = "Resnet"
    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
    CATBOOST = "CatBoost"
    LIGHTGBM = "lightGBM"
    GRADIENT_BOOSTING_TREE = "GradientBoostingTree"
    HIST_GRADIENT_BOOSTING_TREE = "HistGradientBoostingTree"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"
    DECISION_TREE = "DecisionTree"
    KNN = "KNN"
    STG = "STG"
    SVM = "SVM"
    TABNET = "TabNet"
    TABTRANSFORMER = "TabTransformer"
    DEEPFM = "DeepFM"
    VIME = "VIME"
    DANET = "DANet"
    NAM = "NAM"
    NODE = "NODE"


class ModelClass(StrEnum):
    BASE = 'base'
    GBDT = 'GBDT'
    NN = 'NN'
    ICLT = 'ICLT'





class BenchmarkName(StrEnum):
    DEBUG_CATEGORICAL_CLASSIFICATION = "debug_categorical_classification"
    DEBUG_TABZILLA = "debug_tabzilla"

    CATEGORICAL_CLASSIFICATION = "categorical_classification"
    NUMERICAL_CLASSIFICATION = "numerical_classification"
    CATEGORICAL_REGRESSION = "categorical_regression"
    NUMERICAL_REGRESSION = "numerical_regression"
    CATEGORICAL_CLASSIFICATION_LARGE = "categorical_classification_large"
    NUMERICAL_CLASSIFICATION_LARGE = "numerical_classification_large"
    CATEGORICAL_REGRESSION_LARGE = "categorical_regression_large"
    NUMERICAL_REGRESSION_LARGE = "numerical_regression_large"
    
    TABZILLA_HARD = "tabzilla_hard"
    TABZILLA_HARD_MAX_TEN_CLASSES = "tabzilla_hard_max_ten_classes"
    TABZILLA_HAS_COMPLETED_RUNS = "tabzilla_has_completed_runs"


class BenchmarkOrigin(StrEnum):
    TABZILLA = "tabzilla"
    WHYTREES = "whytrees"


class GeneratorName(StrEnum):
    TABPFN = 'tabpfn'
    FOREST = 'forest'
    NEIGHBOR = 'neighbor'


class MetricName(StrEnum):
    ACCURACY = "accuracy"
    F1 = "f1"
    AUC = "auc"
    MSE = "mse"
    R2 = "r2"
    LOG_LOSS = "log_loss"
