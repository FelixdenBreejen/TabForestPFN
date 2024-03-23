from dataclasses import dataclass

from tabularbench.core.enums import BenchmarkOrigin, ModelName


@dataclass
class ConfigPlottingWhytrees():
    n_runs: int
    n_random_shuffles: int
    confidence_bound: float
    plot_default_value: bool
    benchmark_model_names: list[ModelName]

@dataclass
class ConfigPlottingTabzilla():
    benchmark_model_names: list[ModelName]

@dataclass
class ConfigPlotting():
    whytrees: ConfigPlottingWhytrees
    tabzilla: ConfigPlottingTabzilla

    def get_benchmark_model_names(self, benchmark_origin: BenchmarkOrigin):
        match benchmark_origin:
            case BenchmarkOrigin.TABZILLA:
                return self.tabzilla.benchmark_model_names
            case BenchmarkOrigin.WHYTREES:
                return self.whytrees.benchmark_model_names
