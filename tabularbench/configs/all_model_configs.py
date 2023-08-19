import sys
sys.path.append("src")
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from tabularbench.models.skorch_models import create_resnet_skorch, create_ft_transformer_skorch, create_rtdl_mlp_skorch, create_rtdl_mlp_pwl_skorch
from tabularbench.models.skorch_models_regression import create_resnet_regressor_skorch, create_ft_transformer_regressor_skorch, create_rtdl_mlp_regressor_skorch, \
                                            create_rtdl_mlp_pwl_regressor_skorch
from tabularbench.models.torch_models import create_ft_transformer_torch
from tabularbench.models.torch_models_regression import create_ft_transformer_regressor_torch
from tabularbench.models.TabSurvey.models.saint import SAINT



total_config = {}
model_keyword_dic = {}

## ADD YOU MODEL HERE ##
# from tabularbench.configs.model_configs.your_file import config_classif, config_regression, config_classif_default, config_regression_default #replace template.py by your parameters
# keyword = "your_model"
# total_config[keyword] = {
#         "classif": {"random": config_classif,
#                     "default": config_classif_default},
#         "regression": {"random": config_regression,
#                             "default": config_regression_default},
# }
# #these constructor should create an object
# # with fit and predict methods
# model_keyword_dic[config_regression["model_name"]] = YourModelClassRegressor
# model_keyword_dic[config_classif["model_name"]] = YourModelClassClassifier
#############################


from tabularbench.configs.model_configs.gpt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "gbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}
model_keyword_dic[config_regression["model_name"]["value"]] = GradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = GradientBoostingClassifier


from tabularbench.configs.model_configs.rf_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "rf"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = RandomForestRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = RandomForestClassifier

from tabularbench.configs.model_configs.hgbt_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "hgbt"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = HistGradientBoostingRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = HistGradientBoostingClassifier

from tabularbench.configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from tabularbench.configs.model_configs.xgb_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "xgb"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = XGBRegressor
model_keyword_dic[config_classif["model_name"]["value"]] = XGBClassifier

from tabularbench.configs.model_configs.mlp_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "mlp"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_rtdl_mlp_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_rtdl_mlp_skorch

from tabularbench.configs.model_configs.mlp_pwl_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "mlp_pwl"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_rtdl_mlp_pwl_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_rtdl_mlp_pwl_skorch

from tabularbench.configs.model_configs.resnet_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "resnet"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_resnet_regressor_skorch
model_keyword_dic[config_classif["model_name"]["value"]] = create_resnet_skorch

from tabularbench.configs.model_configs.ft_transformer_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "ft_transformer"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = create_ft_transformer_regressor_torch
model_keyword_dic[config_classif["model_name"]["value"]] = create_ft_transformer_torch

from tabularbench.configs.model_configs.saint_config import config_classif, config_regression, config_classif_default, config_regression_default
keyword = "saint"
total_config[keyword] = {
        "classif": {"random": config_classif,
                    "default": config_classif_default},
        "regression": {"random": config_regression,
                            "default": config_regression_default},
}

model_keyword_dic[config_regression["model_name"]["value"]] = SAINT
model_keyword_dic[config_classif["model_name"]["value"]] = SAINT




if __name__ == "__main__":
    print(total_config)
    print(model_keyword_dic)