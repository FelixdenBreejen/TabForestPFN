from dataclasses import dataclass


@dataclass
class ConfigPreprocessing():
    use_quantile_transformer: bool
    use_feature_count_scaling: bool