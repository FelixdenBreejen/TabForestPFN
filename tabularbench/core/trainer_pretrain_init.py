
import torch
from loguru import logger

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.collator import CollatorWithPadding
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.utils.set_seed import seed_worker


def log_parameter_count(cfg: ConfigPretrain, model: torch.nn.Module) -> None:
    
    if cfg.is_main_process:
        logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")


def prepare_ddp_model(cfg: ConfigPretrain, model: torch.nn.Module) -> torch.nn.Module:

    if cfg.use_ddp:
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.device], find_unused_parameters=False)
    
    return model


def create_synthetic_dataset(cfg: ConfigPretrain) -> SyntheticDataset:

    return SyntheticDataset(
        cfg=cfg,
        generator_name=cfg.data.generator,
        min_samples_support=cfg.data.min_samples_support,
        max_samples_support=cfg.data.max_samples_support,
        n_samples_query=cfg.data.n_samples_query,
        min_features=cfg.data.min_features,
        max_features=cfg.data.max_features,
        max_classes=cfg.data.max_classes,
        use_quantile_transformer=cfg.preprocessing.use_quantile_transformer,
        use_feature_count_scaling=cfg.preprocessing.use_feature_count_scaling,
        generator_hyperparams=cfg.data.generator_hyperparams
    )


def create_synthetic_dataloader(cfg: ConfigPretrain, synthetic_dataset: SyntheticDataset) -> torch.utils.data.DataLoader:

    return torch.utils.data.DataLoader(
        synthetic_dataset,
        batch_size=cfg.optim.batch_size,
        collate_fn=CollatorWithPadding(pad_to_n_support_samples=None),
        pin_memory=True,
        num_workers=cfg.workers_per_gpu,
        persistent_workers=cfg.workers_per_gpu > 0,
        worker_init_fn=seed_worker,
    )