import skorch
import numpy as np


class LearningRateLogger(skorch.callbacks.Callback):
    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks

