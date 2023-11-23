from sklearn.base import BaseEstimator
import torch
from torch.optim import AdamW
import numpy as np
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

from tabularbench.core.callbacks import EpochStatistics
from tabularbench.data.dataset_synthetic import SyntheticDataset
from tabularbench.models.tabPFN.tabpfn import TabPFN
from tabularbench.sweeps.config_pretrain import ConfigPretrain



class TrainerPFN(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigPretrain
        ) -> None:

        self.cfg = cfg
        self.model = TabPFN(use_pretrained_weights=False)
        self.model.to(self.cfg.devices[0])   # TODO: DDP

        self.synthetic_dataset = SyntheticDataset(
            min_samples=self.cfg.data['min_samples'],
            max_samples=self.cfg.data['max_samples'],
            min_features=self.cfg.data['min_features'],
            max_features=self.cfg.data['max_features'],
            max_classes=self.cfg.data['max_classes'],
            support_prop=self.cfg.data['support_proportion']
        )

        self.optimizer = self.select_optimizer()
        self.scheduler = self.select_scheduler()
        self.loss = self.select_loss()
        

    def train(self):

        self.model.train()
        generator = self.synthetic_dataset.generator()

        loss_total = 0

        for step in range(1, self.cfg.optim['max_steps']+1):
            dataset = next(generator)

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

            loss = loss.item()
            loss_total += loss

            if step % self.cfg.optim['log_every_n_steps'] == 0:
                self.cfg.logger.info(f"Step {step} | Loss: {loss_total / self.cfg.optim['log_every_n_steps']:.4f}")
                loss_total = 0


    
        if self.cfg['regression']:  
            self.model.decoder[2].reset_parameters()
        
        self.model.to(self.cfg['device'])


        self.x_train = x_train
        self.y_train = y_train

        # if sum(self.cfg['categorical_indicator']) > 0:
            # self.onehot_encoder.fit(self.x_train[:, self.categorical_indicator])
            # self.x_train = self.onehot_encode(self.x_train, max_features=100)

        a = self.make_dataset_split(x_train=x_train, y_train=y_train)
        self.x_train_train, self.x_train_valid, self.y_train_train, self.y_train_valid = a

        dataset_train_generator = TabPFNDatasetGenerator(
            self.x_train_train,
            self.y_train_train,
            regression = self.cfg['regression'],
            batch_size=self.cfg['batch_size'],
        )

        dataset_valid = TabPFNDataset(
            self.x_train_train, 
            self.y_train_train, 
            self.x_train_valid, 
            self.y_train_valid,
            regression = self.cfg['regression'],
            batch_size=self.cfg['batch_size']
        )

        loader_valid = self.make_loader(dataset_valid, training=False)

        self.optimizer = self.select_optimizer()

        self.train(dataset_train_generator, dataset_valid, loader_valid)

        return self


    def train_old(self, dataset_train_generator, dataset_valid, loader_valid):

        for epoch in range(self.cfg['max_epochs']):

            dataset_train = next(dataset_train_generator)            
            loader_train = self.make_loader(dataset_train, training=True)
            
            self.model.train()
        
            epoch_statistics_train = EpochStatistics()

            for batch in loader_train:

                batch = tuple(x.to(self.cfg['device']) for x in batch)
                x_full, y_train, y_test = batch

                y_hat_train = self.model((x_full, y_train), single_eval_pos=dataset_train.single_eval_pos)

                if self.cfg['regression']:
                    y_hat_train = y_hat_train[:, 0]
                else:
                    y_hat_train = y_hat_train[:, :2]
                
                loss = self.loss(y_hat_train, y_test)
                score = self.score(y_hat_train, y_test)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_statistics_train.update(loss.item(), score, len(y_test))

            loss_train, score_train = epoch_statistics_train.get()

            self.model.eval()

            epoch_statistics_valid = EpochStatistics()
            
            with torch.no_grad():

                for batch in loader_valid:

                    batch = tuple(x.to(self.cfg['device']) for x in batch)
                    x_full, y_train, y_test = batch
                    
                    y_hat_valid = self.model((x_full, y_train), single_eval_pos=dataset_valid.single_eval_pos)

                    if self.cfg['regression']:
                        y_hat_valid = y_hat_valid[:, 0]
                    else:
                        y_hat_valid = y_hat_valid[:, :2]

                    loss_valid = self.loss(y_hat_valid, y_test)
                    score_valid = self.score(y_hat_valid, y_test)
                    
                    epoch_statistics_valid.update(loss_valid.item(), score_valid, len(y_test))

            loss_valid, score_valid = epoch_statistics_valid.get()

            print(f"Epoch {epoch} | Train loss: {loss_train:.4f} | Train score: {score_train:.4f} | Valid loss: {loss_valid:.4f} | Valid score: {score_valid:.4f}")

            self.checkpoint(self.model, loss_valid)
            
            self.early_stopping(loss_valid)
            if self.early_stopping.we_should_stop():
                print("Early stopping")
                break

            self.scheduler.step(loss_valid)


    def predict(self, x: np.ndarray):

        self.model.eval()

        # if sum(self.cfg['categorical_indicator']) > 0:
            # x = self.onehot_encode(x, max_features=100)

        y_hat_list = []

        dataset = TabPFNDataset(
            self.x_train, 
            self.y_train, 
            x, 
            regression=self.cfg['regression'],
            batch_size=self.cfg['batch_size']
        )
        loader = self.make_loader(dataset, training=False)

        with torch.no_grad():
            for _ in range(self.cfg['n_ensembles']):
                y_hat_pieces = []

                for batch in loader:
                    
                    batch = tuple(x.to(self.cfg['device']) for x in batch)
                    x_full, y_train, _ = batch

                    output = self.model((x_full, y_train), single_eval_pos=dataset.single_eval_pos)

                    if self.cfg['regression']:
                        output = output[:, 0]
                    else:
                        output = output[:, :2]
                    # output = torch.nn.functional.softmax(output, dim=1)
                    output = output.cpu().numpy()

                    y_hat_pieces.append(output)

                y_hat_list.append(np.concatenate(y_hat_pieces))

        y_hat = sum(y_hat_list) / len(y_hat_list)

        if self.cfg['regression']:
            return y_hat
        else:
            return y_hat.argmax(axis=1)


    def score(self, y_hat, y):

        with torch.no_grad():
            y_hat = y_hat.cpu().numpy()
            y = y.cpu().numpy()

            if self.cfg['regression']:  
                # R2 formula
                ss_res = np.sum((y - y_hat) ** 2, axis=0)
                ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                return r2
            else:
                return np.mean((y_hat.argmax(axis=1) == y))
            

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
            lr=self.cfg.optim['lr'],
            betas=(self.cfg.optim['beta1'], self.cfg.optim['beta2']),
            weight_decay=self.cfg.optim['weight_decay']
        )
        
        return optimizer
        

    def select_scheduler(self):

        if self.cfg.optim['cosine_scheduler']:
            schedule = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.optim['warmup_steps'],
                num_training_steps=self.cfg.optim['max_steps']
            )
        else:
            schedule = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.cfg.optim['warmup_steps']
            )

        return schedule
    

    def select_loss(self):
        return torch.nn.CrossEntropyLoss()
