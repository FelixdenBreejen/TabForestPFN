import torch
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator


class Trainer(BaseEstimator):

    def __init__(self) -> None:
        pass

    def fit():
        pass
        
    def predict():
        pass




class TrainerClassifier(Trainer, ClassifierMixin):
    pass


class TrainerRegressor(Trainer, RegressorMixin):
    pass



