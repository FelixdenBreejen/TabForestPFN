import pandas as pd
from sklearn.base import BaseEstimator
import torch
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from tabularbench.core.collator import collate_with_padding
from tabularbench.core.enums import BenchmarkName, ModelName, SearchType

from tabularbench.core.losses import CrossEntropyLossExtraBatch
from tabularbench.core.metrics import Metrics
from tabularbench.data.benchmarks import BENCHMARKS
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.models.tabPFN.tabpfn import TabPFN
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.config_pretrain import ConfigPretrain
from tabularbench.sweeps.get_logger import get_logger
from tabularbench.sweeps.run_sweep import run_sweep



class TrainerPFN(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain
        ) -> None:

        self.cfg = cfg
        self.model = TabPFN(use_pretrained_weights=False)
        self.model.to(self.cfg.devices[0])   # TODO: DDP

        self.synthetic_dataset = SyntheticDataset(
            cfg=self.cfg,
            min_samples=self.cfg.data.min_samples,
            max_samples=self.cfg.data.max_samples,
            min_features=self.cfg.data.min_features,
            max_features=self.cfg.data.max_features,
            max_classes=self.cfg.data.max_classes,
            support_prop=self.cfg.data.support_proportion
        )
        self.synthetic_dataloader = torch.utils.data.DataLoader(
            self.synthetic_dataset,
            batch_size=self.cfg.optim.batch_size,
            collate_fn=collate_with_padding,
            pin_memory=True
        )


        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        self.loss = self.select_loss()
        

    def train(self):

        self.model.train()
        dataloader = iter(self.synthetic_dataloader)

        metrics = Metrics()

        for step in range(1, self.cfg.optim.max_steps+1):
            dataset = next(dataloader)

            x_support = dataset['x_support'].to(self.cfg.devices[0])
            y_support = dataset['y_support'].to(self.cfg.devices[0])
            x_query = dataset['x_query'].to(self.cfg.devices[0])
            y_query = dataset['y_query'].to(self.cfg.devices[0])

            pred = self.model(x_support, y_support, x_query)
            loss = self.loss(pred, y_query)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            with torch.no_grad():
                metrics.update(pred, y_query)

            if step % self.cfg.optim.log_every_n_steps == 0:
                self.cfg.logger.info(f"Step {step} | Loss: {metrics.loss:.4f} | Accuracy: {metrics.accuracy:.4f}")
                metrics.reset()

            if step % self.cfg.optim.eval_every_n_steps == 0:

                self.model = self.model.to('cpu')
                torch.cuda.empty_cache()

                output_dir = self.cfg.output_dir / f"step_{step}"
                output_dir.mkdir(parents=True, exist_ok=True)

                torch.save(self.model.state_dict(), output_dir / 'model.pt')

                hyperparams_finetuning = self.cfg.hyperparams_finetuning
                hyperparams_finetuning['path_to_weights'] = str(output_dir / 'model.pt')

                model_plot_name = f'TabPFN Reproduction Step {step}'

                cfg_sweep = ConfigBenchmarkSweep(
                    logger=get_logger(output_dir / 'log.txt'),
                    output_dir=output_dir,
                    seed=self.cfg.seed,
                    devices=self.cfg.devices,
                    benchmark=BENCHMARKS[BenchmarkName.CATEGORICAL_CLASSIFICATION],
                    model_name=ModelName.TABPFN_FINETUNE,
                    model_plot_name=model_plot_name,
                    search_type=SearchType.DEFAULT,
                    config_plotting=self.cfg.plotting,
                    n_random_runs_per_dataset=1,
                    n_default_runs_per_dataset=1,
                    openml_dataset_ids_to_ignore=[],
                    hyperparams_object=self.cfg.hyperparams_finetuning
                )
                run_sweep(cfg_sweep)

                default_results = pd.read_csv(output_dir / 'default_results.csv', index_col=0)
                normalized_accuracy = default_results.loc[model_plot_name].iloc[-1]

                self.cfg.logger.info(f"Validation sweep finished")
                self.cfg.logger.info(f"Step {step} | Normalized Validation Accuracy: {normalized_accuracy:.4f}")

                self.model = self.model.to(self.cfg.devices[0])


    def select_optimizer(self):

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
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
