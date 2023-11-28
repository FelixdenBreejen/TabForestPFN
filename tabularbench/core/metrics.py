from pathlib import Path
import einops
from matplotlib import pyplot as plt
import numpy as np
import torch


class MetricsTraining():

    def __init__(self):
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        "Predictions are assumed to be logits"

        pred = einops.rearrange(pred, 'b n c -> (b n) c')
        target = einops.rearrange(target, 'b n -> (b n)')

        loss = torch.nn.functional.cross_entropy(pred, target, reduction='sum')
        self._loss += loss.item()

        pred_ = pred.argmax(dim=-1)
        self._correct += pred_.eq(target).sum().item()

        self._total += target[target != -100].shape[0]

    def reset(self):
        self._loss = 0
        self._correct = 0
        self._total = 0

    
    @property
    def loss(self):
        return self._loss / self._total
    
    @property
    def accuracy(self):
        return self._correct / self._total
    


class MetricsValidation():

    def __init__(self):
        self.reset()

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        "Predictions are assumed to be logits"

        pred = einops.rearrange(pred, 'b n c -> (b n) c')
        target = einops.rearrange(target, 'b n -> (b n)')

        loss = torch.nn.functional.cross_entropy(pred, target, reduction='sum').item()

        pred_ = pred.argmax(dim=-1)
        correct = pred_.eq(target).sum().item()

        total = target[target != -100].shape[0]

        self._loss.append(loss / total)
        self._correct.append(correct / total)


    def update_val(self, norm_acc: float, step: int):
        self._norm_acc.append(norm_acc)
        self._val_step.append(step)



    def reset(self):
        self._loss = []
        self._correct = []
        self._norm_acc = []
        self._val_step = []

    
    def plot(self, output_dir: Path):

        fig, ax = plt.subplots(figsize=(15, 6))

        ax.plot(range(len(self._loss)), self._loss, color='red', label='Cross Entropy Loss (training)')
        ax.set_ylabel('Cross Entropy Loss')

        ax2 = ax.twinx()
        ax2.plot(self._val_step, self._norm_acc, color='blue', label='Normalized Accuracy (validation)')
        ax2.set_ylabel('Normalized accuracy')

        ax2.set_xlabel('Step')

        fig.legend()
        fig.suptitle('PreTraining', fontsize=16)
        
        fig.savefig(output_dir / 'train_plot.png')

        np_dict = {
            'loss': np.array(self._loss),
            'norm_acc': np.array(self._norm_acc),
            'val_step': np.array(self._val_step)
        }
        np.savez(output_dir / 'train_plot.npz', **np_dict)