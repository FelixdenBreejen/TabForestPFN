# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

from xgboost import XGBClassifier, XGBRegressor

from torch import Tensor
import skorch


# %%
class MLP_PWL(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        n_layers: int,
        d_layers: int, #CHANGED
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        feature_importance: np.ndarray,
        d_embedding: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        super().__init__()

        self.tokenizer = TokenizerSquid(d_in, categories, feature_importance, categorical_indicator, d_embedding, True)
        self.d_actual_token = self.tokenizer.d_actual_token

        self.regression = regression
        self.categorical_indicator = categorical_indicator #Added

        d_layers = [d_layers for _ in range(n_layers)] #CHANGED

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i-1] if i > 0 else self.d_actual_token, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):

        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long() #TODO
        else:
            x_num = x
            x_cat = None
        x = []

        x = self.tokenizer(x_num, x_cat)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x

class TokenizerSquid(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        feature_importance: np.ndarray,
        categorical_indicator,
        d_token: int,
        bias: bool,
    ) -> None:
        #categories = None
        super().__init__()


        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)

        d_per_token = d_token
        self.d_tokens = []

        if feature_importance is None:
            self.d_actual_token = d_per_token * d_bias
            return

        self.bias = bias
        feature_importance_cat = feature_importance[categorical_indicator]
        self.bias_cat = nn.ParameterList()

        if categories is not None:  
            self.category_embeddings = nn.ModuleList()     
            for i, cat in enumerate(categories):
                d_token_this = int(d_per_token * feature_importance_cat[i] * d_bias) + 1
                self.d_tokens.append(d_token_this)
                self.category_embeddings.append(nn.Embedding(cat, d_token_this))
                nn_init.kaiming_uniform_(self.category_embeddings[i].weight, a=math.sqrt(5))
                if bias:
                    self.bias_cat.append(nn.Parameter(Tensor(1, d_token_this)))
                if self.bias_cat is not None:
                    nn_init.kaiming_uniform_(self.bias_cat[i], a=math.sqrt(5))
                print(f'{self.category_embeddings[i].weight.shape=}')

        self.weight = nn.ParameterList()
        self.bias_num = nn.ParameterList()
        feature_importance_num = feature_importance[~categorical_indicator]
        
        for i in range(d_numerical):
            d_token_this = int(d_per_token * feature_importance_num[i] * d_bias) + 1
            self.d_tokens.append(d_token_this)
            self.weight.append(nn.Parameter(Tensor(1, d_token_this)))
            if bias:
                self.bias_num.append(nn.Parameter(Tensor(1, d_token_this)))
            # The initialization is inspired by nn.Linear
            nn_init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias_num is not None:
                nn_init.kaiming_uniform_(self.bias_num[i], a=math.sqrt(5))

        self.d_actual_token = sum(self.d_tokens)

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None

        x_num = torch.cat(
            ([] if x_num is None else [x_num]),
            dim=1,
        )
        x_num_parts = torch.split(x_num, 1, dim=1)
        x_parts = []

        for i, x_num_part in enumerate(x_num_parts):
            x_p = self.weight[i][None] * x_num_part[:, :, None]
            if self.bias_num[i] is not None:
                x_p = x_p + self.bias_num[i][None]

            x_parts.append(x_p)

        if x_cat is not None:
            x_cat_parts = torch.split(x_cat, 1, dim=1)
            for i, x_cat_part in enumerate(x_cat_parts):
                x_p = self.category_embeddings[i](x_cat_part)
                if self.bias_cat[i] is not None:
                    x_p = x_p + self.bias_cat[i][None]

                x_parts.append(x_p)        

        x = torch.cat(x_parts, dim=2)
        x = x.reshape(x.shape[0], -1)
        return x    



class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        #categories = None
        super().__init__()


        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)

        d_per_token = d_token
        self.d_actual_token = d_per_token * d_bias

        if categories is not None:        
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_per_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.weight = nn.Parameter(Tensor(d_numerical, d_per_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_per_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            x = x + self.bias[None]

        x = x.reshape(x.shape[0], -1)
        return x


class InputShapeSetterMLP_PWL(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories


    def on_train_begin(self, net, X, y):
        print("categorical_indicator", self.categorical_indicator)
        if self.categorical_indicator is None:
            d_in = X.shape[1]
            categories = None
        else:
            d_in = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                categories = list((X[:, self.categorical_indicator].max(0) + 1).astype(int))
            else:
                categories = self.categories


        path = Path(f'cache/feature_importance_{d_in}_{len(self.categorical_indicator)}.txt')
        if path.exists():
            feature_importance = np.loadtxt(path)
        
        else:
            xgb = XGBClassifier(
                n_estimators=4168,
                eta=0.018,
                subsample=0.839,
                booster='gbtree',
                max_depth=13,
                min_child_weight=2.06,
                colsample_bytree=0.752,
                colsample_bylevel=0.585,
                reg_lambda=0.982,
                reg_alpha=1.113
            )
            xgb.fit(X, y)
            feature_importance = xgb.feature_importances_

            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, feature_importance)
            
        net.set_params(module__d_in=d_in,
                       module__feature_importance=feature_importance,
                       module__categories=categories,  # FIXME #lib.get_categories(X_cat),
                       module__d_out=2 if self.regression == False else 1)  # FIXME#D.info['n_classes'] if D.is_multiclass else 1,
        print("Numerical features: {}".format(d_in))
        print("Categories {}".format(categories))
        print("Feature importance: {}".format(feature_importance))

