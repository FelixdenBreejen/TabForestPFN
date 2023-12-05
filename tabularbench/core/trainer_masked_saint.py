from pathlib import Path

from sklearn.base import BaseEstimator
import torch
from torch.optim import AdamW
import numpy as np
from transformers.optimization import get_cosine_schedule_with_warmup

from tabularbench.core.callbacks import EpochStatistics
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.models.tab_transformer.saint import SAINTadapted


class TrainerMaskedSaint(BaseEstimator):

    def __init__(
            self, 
            cfg: dict
        ) -> None:

        
        self.cfg = cfg

        # self.checkpoint = Checkpoint("temp_weights", self.cfg['id'])

        self.model = SAINTadapted(
            n_classes=self.cfg['data']['max_classes'],
            **self.cfg['model'],
        )
        self.model.to(self.cfg['device'])

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()

        
        self.synthetic_dataset = SyntheticDataset(
            min_samples=self.cfg['data']['min_samples'],
            max_samples=self.cfg['data']['max_samples'],
            min_features=self.cfg['data']['min_features'],
            max_features=self.cfg['data']['max_features'],
            max_classes=self.cfg['data']['max_classes'],
            mask_prop=self.cfg['data']['mask_proportion']
        )
        self.data_loader = iter(self.make_loader(self.synthetic_dataset))

        self.loss = self.select_loss()
        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Training transformer: {num_params/1_000_000:.2f}M parameters")

        self.train()


    def train(self):
            
        self.model.train()

        for epoch in range(self.cfg['max_epochs']):

            epoch_statistics_train = EpochStatistics()

            for step in range(self.cfg['steps_per_epoch']):

                x, x_size_mask, y, y_size_mask, y_label_mask = next(self.data_loader)

                x = x.to(self.cfg['device'])
                x_size_mask = x_size_mask.to(self.cfg['device'])
                y = y.to(self.cfg['device'])
                y_size_mask = y_size_mask.to(self.cfg['device'])
                y_label_mask = y_label_mask.to(self.cfg['device'])
                
                y_logits_all = self.model(x, x_size_mask, y, y_size_mask, y_label_mask)
                y_logits_masked = y_logits_all[y_label_mask]
                y_true_masked = y[y_label_mask]

                loss = self.loss(y_logits_masked, y_true_masked)
                score = self.score(y_logits_masked, y_true_masked)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                epoch_statistics_train.update(loss.item(), score, len(y_true_masked))

            loss_train, score_train = epoch_statistics_train.get()

            print(f"Epoch {epoch} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f}")

            path_weights = Path(self.cfg['output_dir']) / f'epoch_{epoch}_weights.pt'
            torch.save(self.model.state_dict(), path_weights)



    # def predict(self, x: np.ndarray):

    #     self.model.eval()

    #     # if sum(self.cfg['categorical_indicator']) > 0:
    #         # x = self.onehot_encode(x, max_features=100)

    #     y_hat_list = []

    #     dataset = TabPFNDataset(self.x_train, self.y_train, x, batch_size=self.cfg['batch_size'])
    #     loader = self.make_loader(dataset, training=False)

    #     with torch.no_grad():
    #         for _ in range(self.cfg['n_ensembles']):
    #             y_hat_pieces = []

    #             for input in loader:
                    
    #                 input = tuple(x.to(self.cfg['device']) for x in input)
    #                 output = self.model(input, single_eval_pos=dataset.single_eval_pos)
    #                 output = output[:, :2]
    #                 # output = torch.nn.functional.softmax(output, dim=1)
    #                 output = output.cpu().numpy()

    #                 y_hat_pieces.append(output)

    #             y_hat_list.append(np.concatenate(y_hat_pieces))

    #     y_hat = sum(y_hat_list) / len(y_hat_list)

    #     if self.cfg['regression']:
    #         return y_hat
    #     else:
    #         return y_hat.argmax(axis=1)


    def score(self, y_hat, y):
        with torch.no_grad():
            return np.mean((y_hat.cpu().numpy().argmax(axis=1) == y.cpu().numpy()))
            

    def load_params(self, path):
        self.model.load_state_dict(torch.load(path))

    
    def onehot_encode(self, x: np.ndarray, max_features: int):

        x_categorical = self.onehot_encoder.transform(x[:, self.categorical_indicator])
        x_numerical = x[:, ~self.categorical_indicator]

        x_new = np.concatenate([x_numerical, x_categorical], axis=1)

        if x_new.shape[1] < max_features:
            return x_new
        else:
            print("Warning: onehot-encoded features exceed max_features. Returning original features.")
            return x


    def select_optimizer(self):

        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.cfg['lr'],
            betas=(self.cfg['beta1'], self.cfg['beta2']),
            weight_decay=self.cfg['weight_decay']
        )
        
        return optimizer
        

    def select_scheduler(self):

        steps_total = self.cfg['steps_per_epoch'] * self.cfg['max_epochs']

        if self.cfg['lr_scheduler']:                
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=steps_total * 0.01,
                num_training_steps=steps_total
            )
        else:
            # Identity scheduler
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)

        return scheduler
    
        

    def make_loader(self, dataset):

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['batch_size'],
            pin_memory=True,
        )

    
    def select_loss(self):

        loss = torch.nn.CrossEntropyLoss()

        return loss
    




