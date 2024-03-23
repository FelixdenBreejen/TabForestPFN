from pathlib import Path

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.enums import BenchmarkOrigin, ModelName, Phase, SearchType
from tabularbench.data.benchmarks import Benchmark


def create_config_benchmark_sweep(
        cfg: ConfigPretrain,
        benchmark: Benchmark, 
        output_dir: Path,
        weights_path: Path,
        plot_name: str,
        phase: Phase,
    ) -> ConfigBenchmarkSweep:

    hyperparams_finetuning = make_hyperparams_finetuning_dict(cfg, weights_path)

    return ConfigBenchmarkSweep(
        output_dir=output_dir,
        seed=cfg.seed,
        devices=cfg.devices,
        benchmark=benchmark,
        model_name=cfg.model_name,
        model_plot_name=plot_name,
        search_type=SearchType.DEFAULT,
        plotting=cfg.plotting,
        n_random_runs_per_dataset=1,
        n_default_runs_per_dataset=decide_n_default_runs_per_dataset(cfg, benchmark, phase),
        openml_dataset_ids_to_ignore=cfg.testing.openml_dataset_ids_to_ignore,
        hyperparams_object=hyperparams_finetuning
    )


def make_hyperparams_finetuning_dict(cfg: ConfigPretrain, weights_path: Path) -> dict:

    hyperparams_finetuning = cfg.hyperparams_finetuning
    hyperparams_finetuning['use_pretrained_weights'] = True
    hyperparams_finetuning['path_to_weights'] = str(weights_path)
    hyperparams_finetuning['use_quantile_transformer'] = cfg.preprocessing.use_quantile_transformer
    hyperparams_finetuning['use_feature_count_scaling'] = cfg.preprocessing.use_feature_count_scaling

    if cfg.model_name == ModelName.FOUNDATION:
        hyperparams_finetuning['n_features'] = cfg.data.max_features
        hyperparams_finetuning['n_classes'] = cfg.data.max_classes

        for key, value in cfg.model.items():
            hyperparams_finetuning[key] = value

    return hyperparams_finetuning


def decide_n_default_runs_per_dataset(cfg: ConfigPretrain, benchmark: Benchmark, phase: Phase) -> int:

    match (benchmark.origin, phase):
        case (BenchmarkOrigin.WHYTREES, Phase.VALIDATION):
            return cfg.testing.n_default_runs_per_dataset_valid
        case (BenchmarkOrigin.WHYTREES, Phase.TESTING):
            return cfg.testing.n_default_runs_per_dataset_test
        case (BenchmarkOrigin.TABZILLA, _):
            return 1
        case (_, _):
            raise ValueError("Not decided how to set the number of runs")