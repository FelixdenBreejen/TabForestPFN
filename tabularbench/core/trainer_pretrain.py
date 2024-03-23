from pathlib import Path

import pandas as pd
import torch
import torch.multiprocessing as mp
from loguru import logger
from sklearn.base import BaseEstimator

from tabularbench.config.config_pretrain import ConfigPretrain
from tabularbench.core.enums import BenchmarkName, DataSplit, Phase
from tabularbench.core.get_model import get_model_pretrain
from tabularbench.core.get_optimizer import get_optimizer_pretrain
from tabularbench.core.get_scheduler import get_scheduler_pretrain
from tabularbench.core.losses import CrossEntropyLossExtraBatch
from tabularbench.core.metrics import MetricsTraining, MetricsValidation
from tabularbench.core.trainer_pretrain_evaluate import create_config_benchmark_sweep
from tabularbench.core.trainer_pretrain_init import (create_synthetic_dataloader, create_synthetic_dataset,
                                                     log_parameter_count, prepare_ddp_model)
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.utils.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME


class TrainerPretrain(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain,
            barrier: mp.Barrier
        ) -> None:

        self.cfg = cfg
        self.barrier = barrier
        self.model_ = get_model_pretrain(cfg)
        self.model_.to(cfg.device)

        log_parameter_count(cfg, self.model_)
        self.model = prepare_ddp_model(cfg, self.model_)

        self.synthetic_dataset = create_synthetic_dataset(cfg)
        self.synthetic_dataloader = create_synthetic_dataloader(cfg, self.synthetic_dataset)

        self.optimizer = get_optimizer_pretrain(cfg, self.model)
        self.scheduler = get_scheduler_pretrain(cfg, self.optimizer)
        self.loss = CrossEntropyLossExtraBatch()

        self.metrics_train = MetricsTraining()
        self.metrics_val = MetricsValidation()

        self.step = 0
        self.dataloader = None
        

    def train(self):

        self.model.train()
        self.dataloader = iter(self.synthetic_dataloader)

        for step in range(1, self.cfg.optim.max_steps+1):
            self.step = step
            self.train_one_step()

    
    def train_one_step(self):
            
        pred, y_query = self.process_next_batch()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.update_metrics(pred, y_query)

        if self.log_this_step() and self.cfg.is_main_process:
            self.log_training_metrics()

        if self.eval_this_step():
            self.move_model_to_cpu()
            self.evaluate_current_model()
            self.wait_and_move_model_to_gpu()


    def process_next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:

        preds = []
        y_queries = []

        for _ in range(self.cfg.optim.gradient_accumulation_steps):

            dataset = next(self.dataloader)

            x_support = dataset['x_support'].to(self.cfg.device)
            y_support = dataset['y_support'].to(self.cfg.device)
            x_query = dataset['x_query'].to(self.cfg.device)
            y_query = dataset['y_query'].to(self.cfg.device)

            pred = self.model(x_support, y_support, x_query)
            loss = self.loss(pred, y_query)
            
            loss = loss / self.cfg.optim.gradient_accumulation_steps
            loss.backward()

            preds.append(pred.detach().cpu())
            y_queries.append(y_query.detach().cpu())

        return torch.cat(preds), torch.cat(y_queries)
    

    def evaluate_current_model(self):

        if not self.cfg.is_main_process:
            return

        logger.info(f"Starting validation sweep")

        output_dir, weights_path = self.prepare_directories_and_weights()

        normalized_accuracies = self.validate(output_dir, weights_path, plot_name=f"{self.cfg.model_name.value} Pretrain Step {self.step}")

        logger.info(f"Finished validation sweep")
        logger.info(f"Normalized Validation Accuracy: {normalized_accuracies[DataSplit.VALID]:.4f}")
        logger.info(f"Normalized Test Accuracy: {normalized_accuracies[DataSplit.TEST]:.4f}")

        self.metrics_val.update_val(normalized_accuracies[DataSplit.VALID], normalized_accuracies[DataSplit.TEST], self.step)
        self.metrics_val.plot(self.cfg.output_dir)


    def log_this_step(self):
        return self.step % self.cfg.optim.log_every_n_steps == 0
    

    def log_training_metrics(self):
        logger.info(f"Step {self.step} | Loss: {self.metrics_train.loss:.4f} | Accuracy: {self.metrics_train.accuracy:.4f}")
        self.metrics_train.reset()


    def eval_this_step(self):
        return self.step % self.cfg.optim.eval_every_n_steps == 0


    def update_metrics(self, pred: torch.Tensor, y_query: torch.Tensor):
        
        with torch.no_grad():
            self.metrics_train.update(pred, y_query)
            self.metrics_val.update(pred, y_query)


    def move_model_to_cpu(self):

        del self.model
        self.model_.cpu()
        torch.cuda.empty_cache()

        # wait until all gpus have moved the model to cpu
        torch.distributed.barrier()


    def wait_and_move_model_to_gpu(self):

        # We cannot use the torch distributed barrier here, because that blocks the execution on the gpus.
        # This barrier only blocks execution on the cpu of the current process, which doesn't interfere with the validation sweep.
        self.barrier.wait()

        # see https://github.com/pytorch/pytorch/issues/104336
        self.model_.to(self.cfg.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model_, device_ids=[self.cfg.device], find_unused_parameters=False)


    def prepare_directories_and_weights(self) -> tuple[Path, Path]:

        weights_path = self.cfg.output_dir / 'weights' / f"model_step_{self.step}.pt"
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_weights_path = weights_path

        output_dir = self.cfg.output_dir / f"step_{self.step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = self.model_.state_dict()
        torch.save(state_dict, weights_path)

        return output_dir, weights_path


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


    