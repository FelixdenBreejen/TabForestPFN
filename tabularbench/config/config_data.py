from dataclasses import dataclass

from tabularbench.core.enums import GeneratorName


@dataclass
class ConfigData():
    generator: GeneratorName
    min_samples_support: int
    max_samples_support: int
    n_samples_query: int
    min_features: int
    max_features: int
    max_classes: int
    generator_hyperparams: dict