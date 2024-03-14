
from tabularbench.core.enums import ModelClass, ModelName


def get_model_class(model_name: ModelName) -> ModelClass:
    match model_name:
        case ModelName.TABPFN | ModelName.FOUNDATION:
            return ModelClass.ICLT
        case ModelName.RANDOM_FOREST | ModelName.SVM | ModelName.KNN | ModelName.DECISION_TREE | ModelName.LINEAR_REGRESSION:
            return ModelClass.BASE
        case ModelName.CATBOOST | ModelName.LIGHTGBM | ModelName.GRADIENT_BOOSTING_TREE | ModelName.HIST_GRADIENT_BOOSTING_TREE | ModelName.XGBOOST:
            return ModelClass.GBDT
        case ModelName.FT_TRANSFORMER | ModelName.SAINT | ModelName.MLP | ModelName.RESNET | ModelName.TABNET | ModelName.MLP_RTDL \
           | ModelName.TABTRANSFORMER | ModelName.DEEPFM | ModelName.VIME | ModelName.DANET | ModelName.NAM | ModelName.NODE | ModelName.STG:
            return ModelClass.NN
        case _:
            raise ValueError(f"Model {model_name} not found in ModelClass enum")