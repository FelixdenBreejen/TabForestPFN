from pathlib import Path
import pandas as pd
from sklearn.base import BaseEstimator
import torch
import torch.multiprocessing as mp
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from tabularbench.core.collator import collate_with_padding
from tabularbench.core.enums import BenchmarkName, DataSplit, ModelName, SearchType
from tabularbench.core.losses import CrossEntropyLossExtraBatch
from tabularbench.core.metrics import MetricsTraining, MetricsValidation
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.models.tabPFN.tabpfn_transformer import TabPFN
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.config_pretrain import ConfigPretrain
from tabularbench.sweeps.get_logger import get_logger
from tabularbench.sweeps.paths_and_filenames import DEFAULT_RESULTS_TEST_FILE_NAME, DEFAULT_RESULTS_VAL_FILE_NAME
from tabularbench.sweeps.run_sweep import run_sweep
from tabularbench.sweeps.set_seed import seed_worker



class TrainerPFN(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain,
            barrier: mp.Barrier
        ) -> None:

        self.cfg = cfg
        self.barrier = barrier
        self.model = TabPFN(use_pretrained_weights=False)
        self.model.to(self.cfg.device)

        if cfg.use_ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.device], find_unused_parameters=False)

        self.synthetic_dataset = SyntheticDataset(
            cfg=self.cfg,
            min_samples_support=self.cfg.data.min_samples_support,
            max_samples_support=self.cfg.data.max_samples_support,
            n_samples_query=self.cfg.data.n_samples_query,
            min_features=self.cfg.data.min_features,
            max_features=self.cfg.data.max_features,
            max_classes=self.cfg.data.max_classes
        )

        self.synthetic_dataloader = torch.utils.data.DataLoader(
            self.synthetic_dataset,
            batch_size=self.cfg.optim.batch_size,
            collate_fn=collate_with_padding,
            pin_memory=True,
            num_workers=self.cfg.workers_per_gpu,
            persistent_workers=self.cfg.workers_per_gpu > 0,
            worker_init_fn=seed_worker,
        )


        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        self.loss = self.select_loss()
        

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
                self.cfg.logger.info(f"Step {step} | Loss: {metrics_train.loss:.4f} | Accuracy: {metrics_train.accuracy:.4f}")
                metrics_train.reset()

            if step % self.cfg.optim.eval_every_n_steps == 0:
                self.model = self.model.to('cpu')
                torch.cuda.empty_cache()

                torch.distributed.barrier()

                if self.cfg.is_main_process:

                    self.cfg.logger.info(f"Starting validation sweep")

                    weights_path = self.cfg.output_dir / 'weights' / f"model_step_{step}.pt"
                    weights_path.parent.mkdir(parents=True, exist_ok=True)
                    self.last_weights_path = weights_path

                    output_dir = self.cfg.output_dir / f"step_{step}"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    state_dict = { k.replace('module.', ''): v for k, v in self.model.state_dict().items()  }
                    torch.save(state_dict, weights_path)

                    normalized_accuracies = self.validate(output_dir, weights_path, plot_name=f"TabPFN Reproduction Step {step}")

                    self.cfg.logger.info(f"Finished validation sweep")
                    self.cfg.logger.info(f"Normalized Validation Accuracy: {normalized_accuracies[DataSplit.VALID]:.4f}")
                    self.cfg.logger.info(f"Normalized Test Accuracy: {normalized_accuracies[DataSplit.TEST]:.4f}")

                    metrics_val.update_val(normalized_accuracies[DataSplit.VALID], normalized_accuracies[DataSplit.TEST], step)
                    metrics_val.plot(self.cfg.output_dir)


                # We cannot use the torch distributed barrier here, because that blocks the execution on the gpus.
                # This barrier only blocks execution on the cpu of the current process, which doesn't interfere with the validation sweep.
                self.barrier.wait()

                self.model = self.model.to(self.cfg.device)


    def validate(self, output_dir: Path, weights_path: Path, plot_name: str) -> dict[DataSplit, float]:

        hyperparams_finetuning = self.cfg.hyperparams_finetuning
        hyperparams_finetuning['path_to_weights'] = str(weights_path)

        cfg_sweep = ConfigBenchmarkSweep(
            logger=get_logger(output_dir / 'log.txt'),
            output_dir=output_dir,
            seed=self.cfg.seed,
            devices=self.cfg.devices,
            benchmark=BENCHMARKS[BenchmarkName.CATEGORICAL_CLASSIFICATION],
            model_name=ModelName.TABPFN_FINETUNE,
            model_plot_name=plot_name,
            search_type=SearchType.DEFAULT,
            config_plotting=self.cfg.plotting,
            n_random_runs_per_dataset=1,
            n_default_runs_per_dataset=1,
            openml_dataset_ids_to_ignore=[],
            hyperparams_object=self.cfg.hyperparams_finetuning
        )
        run_sweep(cfg_sweep)

        default_results_val = pd.read_csv(output_dir / DEFAULT_RESULTS_VAL_FILE_NAME, index_col=0)
        normalized_accuracy_val = default_results_val.loc[plot_name].iloc[-1]

        default_results_test = pd.read_csv(output_dir / DEFAULT_RESULTS_TEST_FILE_NAME, index_col=0)
        normalized_accuracy_test = default_results_test.loc[plot_name].iloc[-1]

        return {
            DataSplit.VALID: normalized_accuracy_val,
            DataSplit.TEST: normalized_accuracy_test
        }
    

    def test(self):

        weights_path = self.last_weights_path
        output_dir = self.cfg.output_dir / 'test'
        plot_name = f"TabPFN Reproduction Test"

        hyperparams_finetuning = self.cfg.hyperparams_finetuning
        hyperparams_finetuning['path_to_weights'] = str(weights_path)

        cfg_sweep = ConfigBenchmarkSweep(
            logger=get_logger(output_dir / 'log.txt'),
            output_dir=output_dir,
            seed=self.cfg.seed,
            devices=self.cfg.devices,
            benchmark=BENCHMARKS[BenchmarkName.CATEGORICAL_CLASSIFICATION],
            model_name=ModelName.TABPFN_FINETUNE,
            model_plot_name=plot_name,
            search_type=SearchType.DEFAULT,
            config_plotting=self.cfg.plotting,
            n_random_runs_per_dataset=1,
            n_default_runs_per_dataset=10,
            openml_dataset_ids_to_ignore=[],
            hyperparams_object=self.cfg.hyperparams_finetuning
        )
        run_sweep(cfg_sweep)











    def select_optimizer(self):

        parameters = [(name, param) for name, param in self.model.named_parameters()]

        parameters_with_weight_decay = []
        parameters_without_weight_decay = []

        for name, param in parameters:
            if name.endswith(".bias") or '.norm' in name:
                parameters_without_weight_decay.append(param)
            else:
                parameters_with_weight_decay.append(param)

        optimizer_parameters = [
            {"params": parameters_with_weight_decay, "weight_decay": self.cfg.optim.weight_decay},
            {"params": parameters_without_weight_decay, "weight_decay": 0.0},
        ]
    
        optimizer = torch.optim.Adam(
            optimizer_parameters, 
            lr=self.cfg.optim.lr,
            betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
            weight_decay=self.cfg.optim.weight_decay
        )
        
        return optimizer
        

    def select_scheduler(self):

        if self.cfg.optim.cosine_scheduler:
            schedule = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.optim.warmup_steps,
                num_training_steps=self.cfg.optim.max_steps
            )
        else:
            schedule = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.optim.warmup_steps
            )

        return schedule
    

    def select_loss(self):
        return CrossEntropyLossExtraBatch()
