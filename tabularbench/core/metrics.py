import einops
import torch


class Metrics():

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