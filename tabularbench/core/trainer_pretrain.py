from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from sklearn.base import BaseEstimator

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.collator import CollatorWithPadding
from tabularbench.core.enums import BenchmarkName, DataSplit, Phase
from tabularbench.core.get_model import get_model_pretrain
from tabularbench.core.get_optimizer import get_optimizer_pretrain
from tabularbench.core.get_scheduler import get_scheduler_pretrain
from tabularbench.core.losses import CrossEntropyLossExtraBatch
from tabularbench.core.metrics import MetricsTraining, MetricsValidation
from tabularbench.core.trainer_pretrain_evaluate import create_config_benchmark_sweep
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME
from tabularbench.utils.set_seed import seed_worker


class TrainerPretrain(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain,
            barrier: mp.Barrier
        ) -> None:

        self.cfg = cfg
        self.barrier = barrier
        self.model_ = get_model_pretrain(cfg)
        self.model_.to(self.cfg.device)

        if cfg.is_main_process:
            logger.info(f"Model has {sum(p.numel() for p in self.model_.parameters() if p.requires_grad):,} trainable parameters")

        if cfg.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model_, device_ids=[cfg.device], find_unused_parameters=False)
        else:
            self.model = self.model_

        self.synthetic_dataset = SyntheticDataset(
            cfg=self.cfg,
            generator_name=self.cfg.data.generator,
            min_samples_support=self.cfg.data.min_samples_support,
            max_samples_support=self.cfg.data.max_samples_support,
            n_samples_query=self.cfg.data.n_samples_query,
            min_features=self.cfg.data.min_features,
            max_features=self.cfg.data.max_features,
            max_classes=self.cfg.data.max_classes,
            use_quantile_transformer=self.cfg.preprocessing.use_quantile_transformer,
            use_feature_count_scaling=self.cfg.preprocessing.use_feature_count_scaling,
            generator_hyperparams=self.cfg.data.generator_hyperparams
        )

        self.synthetic_dataloader = torch.utils.data.DataLoader(
            self.synthetic_dataset,
            batch_size=self.cfg.optim.batch_size,
            collate_fn=CollatorWithPadding(pad_to_n_support_samples=None),
            pin_memory=True,
            num_workers=self.cfg.workers_per_gpu,
            persistent_workers=self.cfg.workers_per_gpu > 0,
            worker_init_fn=seed_worker,
        )


        self.optimizer = get_optimizer_pretrain(cfg, self.model)
        self.scheduler = get_scheduler_pretrain(cfg, self.optimizer)
        self.loss = CrossEntropyLossExtraBatch()
        

    def train(self):

        self.model.train()
        dataloader = iter(self.synthetic_dataloader)

        metrics_train = MetricsTraining()
        metrics_val = MetricsValidation()

        for step in range(1, self.cfg.optim.max_steps+1):
            
            for _ in range(self.cfg.optim.gradient_accumulation_steps):

                dataset = next(dataloader)

                x_support = dataset['x_support'].to(self.cfg.device)
                y_support = dataset['y_support'].to(self.cfg.device)
                x_query = dataset['x_query'].to(self.cfg.device)
                y_query = dataset['y_query'].to(self.cfg.device)

                pred = self.model(x_support, y_support, x_query)
                loss = self.loss(pred, y_query)
                
                loss = loss / self.cfg.optim.gradient_accumulation_steps
                loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()


            with torch.no_grad():
                metrics_train.update(pred, y_query)
                metrics_val.update(pred, y_query)

            if step % self.cfg.optim.log_every_n_steps == 0 and self.cfg.is_main_process:
                logger.info(f"Step {step} | Loss: {metrics_train.loss:.4f} | Accuracy: {metrics_train.accuracy:.4f}")
                metrics_train.reset()

            if step % self.cfg.optim.eval_every_n_steps == 0:
                
                del self.model
                self.model_.cpu()
                torch.cuda.empty_cache()

                # wait until all gpus have moved the model to cpu
                torch.distributed.barrier()

                if self.cfg.is_main_process:

                    logger.info(f"Starting validation sweep")

                    weights_path = self.cfg.output_dir / 'weights' / f"model_step_{step}.pt"
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    self.last_weights_path = weights_path

                    output_dir = self.cfg.output_dir / f"step_{step}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    state_dict = self.model_.state_dict()
                    torch.save(state_dict, weights_path)

                    normalized_accuracies = self.validate(output_dir, weights_path, plot_name=f"{self.cfg.model_name.value} Pretrain Step {step}")

                    logger.info(f"Finished validation sweep")
                    logger.info(f"Normalized Validation Accuracy: {normalized_accuracies[DataSplit.VALID]:.4f}")
                    logger.info(f"Normalized Test Accuracy: {normalized_accuracies[DataSplit.TEST]:.4f}")

                    metrics_val.update_val(normalized_accuracies[DataSplit.VALID], normalized_accuracies[DataSplit.TEST], step)
                    metrics_val.plot(self.cfg.output_dir)


                # We cannot use the torch distributed barrier here, because that blocks the execution on the gpus.
                # This barrier only blocks execution on the cpu of the current process, which doesn't interfere with the validation sweep.
                self.barrier.wait()

                # see https://github.com/pytorch/pytorch/issues/104336
                self.model_.to(self.cfg.device)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model_, device_ids=[self.cfg.device], find_unused_parameters=False)


    def validate(self, output_dir: Path, weights_path: Path, plot_name: str) -> dict[DataSplit, float]:

        cfg_sweep = create_config_benchmark_sweep(
            cfg=self.cfg,
            benchmark=BENCHMARKS[BenchmarkName.CATEGORICAL_CLASSIFICATION],
            output_dir=output_dir,
            weights_path=weights_path,
            plot_name=plot_name,
            phase=Phase.VALIDATION
        )
        run_sweep(cfg_sweep)

        default_results_val = pd.read_csv(output_dir / DEFAULT_RESULTS_VAL_FILE_NAME, index_col=0)
        normalized_score_val = default_results_val.loc[plot_name].iloc[-1]

        default_results_test = pd.read_csv(output_dir / DEFAULT_RESULTS_TEST_FILE_NAME, index_col=0)
        normalized_score_test = default_results_test.loc[plot_name].iloc[-1]

        return {
            DataSplit.VALID: normalized_score_val,
            DataSplit.TEST: normalized_score_test
        }
    

    def test(self):
        
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()

        for benchmark_name in self.cfg.testing.benchmarks:
                
            benchmark = BENCHMARKS[benchmark_name]
            output_dir = self.cfg.output_dir / f"test_{benchmark_name.value}"

            cfg_sweep = create_config_benchmark_sweep(
                cfg=self.cfg,
                benchmark=benchmark,
                output_dir=output_dir,
                weights_path=self.last_weights_path,
                plot_name=f"{self.cfg.model_name.value} Pretrain Test",
                phase=Phase.TESTING
            )
            run_sweep(cfg_sweep)

