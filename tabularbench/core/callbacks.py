"""
These are not callbacks, but if you look for callback most likely you look for these
"""

from pathlib import Path
import torch
import numpy as np


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

    def __init__(self, dirname, id):
        self.dirname = dirname
        self.id = id
        self.curr_best_loss = np.inf
        

    def __call__(self, net, loss):
        
        if loss < self.curr_best_loss:
            self.curr_best_loss = loss
            path = Path(self.dirname) / f"params_{self.id}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(net.state_dict(), path)