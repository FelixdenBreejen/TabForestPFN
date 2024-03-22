from tabularbench.core.enums import GeneratorName
from tabularbench.data.synthetic_generator_forest import synthetic_dataset_generator_forest
from tabularbench.data.synthetic_generator_mix import synthetic_dataset_generator_mix
from tabularbench.data.synthetic_generator_neighbor import synthetic_dataset_generator_neighbor
from tabularbench.data.synthetic_generator_tabpfn import synthetic_dataset_generator_tabpfn


class SyntheticDatasetGeneratorSelectorMixin():

    def select_synthetic_dataset_generator(self):

        match self.generator_name:
            case GeneratorName.TABPFN:
                return synthetic_dataset_generator_tabpfn(
                    n_samples=self.n_samples,
                    min_features=self.min_features,
                    max_features=self.max_features,
                    max_classes=self.max_classes
                )
            case GeneratorName.FOREST:
                return synthetic_dataset_generator_forest(
                    n_samples=self.n_samples,
                    min_features=self.min_features,
                    max_features=self.max_features,
                    max_classes=self.max_classes,
                    base_size=self.generator_hyperparams['base_size'],
                    min_depth=self.generator_hyperparams['min_depth'],
                    max_depth=self.generator_hyperparams['max_depth'],
                    categorical_x=self.generator_hyperparams['categorical_x'],
                )
            case GeneratorName.NEIGHBOR:
                return synthetic_dataset_generator_neighbor(
                    n_samples=self.n_samples,
                    min_features=self.min_features,
                    max_features=self.max_features,
                    max_classes=self.max_classes,
                    min_neighbors=self.generator_hyperparams['min_neighbors'],
                    max_neighbors=self.generator_hyperparams['max_neighbors'],
                )
            case GeneratorName.MIX:
                return synthetic_dataset_generator_mix(
                    n_samples=self.n_samples,
                    min_features=self.min_features,
                    max_features=self.max_features,
                    max_classes=self.max_classes,
                    base_size=self.generator_hyperparams['base_size'],
                    min_depth=self.generator_hyperparams['min_depth'],
                    max_depth=self.generator_hyperparams['max_depth'],
                    categorical_x=self.generator_hyperparams['categorical_x'],
                )