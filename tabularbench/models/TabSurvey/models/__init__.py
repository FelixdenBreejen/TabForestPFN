all_models = ["LinearModel", "KNN", "DecisionTree", "RandomForest", "XGBoost", "CatBoost", "LightGBM", "ModelTree",
               "MLP", "TabNet", "VIME", "TabTransformer", "NODE", "DeepGBM", "RLN", "DNFNet", "STG", "NAM", "DeepFM",
               "SAINT", "DANet"]


def str2model(model):

    if model == "LinearModel":
        from tabularbench.models.baseline_models import LinearModel
        return LinearModel

    elif model == "KNN":
        from tabularbench.models.baseline_models import KNN
        return KNN

    elif model == "SVM":
        from tabularbench.models.baseline_models import SVM
        return SVM

    elif model == "DecisionTree":
        from tabularbench.models.baseline_models import DecisionTree
        return DecisionTree

    elif model == "RandomForest":
        from tabularbench.models.baseline_models import RandomForest
        return RandomForest

    elif model == "XGBoost":
        from tabularbench.models.tree_models import XGBoost
        return XGBoost

    elif model == "CatBoost":
        from tabularbench.models.tree_models import CatBoost
        return CatBoost

    elif model == "LightGBM":
        from tabularbench.models.tree_models import LightGBM
        return LightGBM

    elif model == "MLP":
        from tabularbench.models.mlp import MLP
        return MLP

    elif model == "ModelTree":
        from tabularbench.models.modeltree import ModelTree
        return ModelTree

    elif model == "TabNet":
        from tabularbench.models.tabnet import TabNet
        return TabNet

    elif model == "VIME":
        from tabularbench.models.vime import VIME
        return VIME

    elif model == "TabTransformer":
        from tabularbench.models.tabtransformer import TabTransformer
        return TabTransformer

    elif model == "NODE":
        from tabularbench.models.node import NODE
        return NODE

    elif model == "DeepGBM":
        from tabularbench.models.deepgbm import DeepGBM
        return DeepGBM

    elif model == "RLN":
        from tabularbench.models.rln import RLN
        return RLN

    elif model == "DNFNet":
        from tabularbench.models.dnf import DNFNet
        return DNFNet

    elif model == "STG":
        from tabularbench.models.stochastic_gates import STG
        return STG

    elif model == "NAM":
        from tabularbench.models.neural_additive_models import NAM
        return NAM

    elif model == "DeepFM":
        from tabularbench.models.deepfm import DeepFM
        return DeepFM

    elif model == "SAINT":
        from tabularbench.models.saint import SAINT
        return SAINT

    elif model == "DANet":
        from tabularbench.models.danet import DANet
        return DANet

    else:
        raise NotImplementedError("Model \"" + model + "\" not yet implemented")
