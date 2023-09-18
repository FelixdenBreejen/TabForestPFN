from enum import Enum, IntEnum


class Task(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class FeatureType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2
    MIXED = 3


class SearchType(Enum):
    DEFAULT = 1
    RANDOM = 2


class DatasetSize(IntEnum):
    SMALL = 1000
    MEDIUM = 10000
    LARGE = 50000


class ModelName(Enum):
    FT_TRANSFORMER = 1
    TABPFN_FINETUNE = 2

