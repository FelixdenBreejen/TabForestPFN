"""
These are not callbacks, but if you look for callback most likely you look for these
"""

from pathlib import Path

import numpy as np
import torch


class EarlyStopping():

    def __init__(self, patience=10, delta=0.0001):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta


    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def we_should_stop(self):
        return self.early_stop


class Checkpoint():

    def __init__(self, dirname: Path, id: str):
        self.dirname = dirname
        self.id = id
        self.curr_best_loss = np.inf
        self.path = Path(self.dirname) / f"params_{self.id}.pt"

    
    def reset(self, net):
        self.curr_best_loss = np.inf
        torch.save(net.state_dict(), self.path)
        

    def __call__(self, net, loss):
        
        if loss < self.curr_best_loss:
            self.curr_best_loss = loss
            self.path.parent.mkdir(exist_ok=True)
            torch.save(net.state_dict(), self.path)



class EpochStatistics():

    def __init__(self) -> None:
        self.n = 0
        self.loss = 0
        self.score = 0
        
    def update(self, loss, score, n):
        self.n += n
        self.loss += loss * n
        self.score += score * n

    def get(self):
        return self.loss / self.n, self.score / self.n
    


class TrackOutput():

    def __init__(self) -> None:
        self.y_true = []
        self.y_pred = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        assert len(y_true.shape) == len(y_pred.shape) == 1
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def get(self):
        return np.concatenate(self.y_true, axis=0), np.concatenate(self.y_pred, axis=0)