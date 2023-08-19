from __future__ import annotations
import math
import typing as ty

import scipy
import skorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import sys
sys.path.append("")
import tabularbench.models.tabular.lib as lib


# %%
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
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
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
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_qkv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_qkv), self.W_k(x_qkv), self.W_v(x_qkv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)

        x = attention @ self._reshape(v)
        
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        feature_representation_list: FeatureRepresentationList,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        print("d_token {}".format(d_token))

        self.categorical_indicator = categorical_indicator
        self.regression = regression

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x) -> Tensor:
        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long() #TODO
        else:
            x_num = x
            x_cat = None
        #x_cat = None #FIXME
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x_residual = x_residual[:, :1] if is_last_layer else x_residual

            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)


        return x    


class InputShapeSetterTransformer(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        print("categorical_indicator", self.categorical_indicator)
        if self.categorical_indicator is None:
            d_numerical = X.shape[1]
            categories = None
        else:
            d_numerical = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                categories = list((X[:, self.categorical_indicator].max(0) + 1).astype(int))
            else:
                categories = self.categories

        feature_representation = FeatureRepresentationList.create_representations("quantile", 10, X[:, ~self.categorical_indicator])

        print("Numerical features: {}".format(d_numerical))
        print("Categories {}".format(categories))

        return {
            'module__d_numerical': d_numerical,
            'module__categories': categories,
            'module__feature_representation_list': feature_representation,
            'module__d_out': 2 if self.regression == False else 1
        }





class FeatureRepresentation(np.ndarray):
    """
    A feature representation is a summary of a feature in a dataset.
    It is a 1D numpy array of shape (n), where n is the size of the feature representation.
    The feature representation should have a significant smaller size than the original feature.
    Also, all values in the feature representation should be unique.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def get_values(self) -> np.ndarray:
        """
        Return the values of the feature representation.
        """
        return self


    def get_bounds(self, add_inf: bool = False) -> np.ndarray:
        """
        In case we have the array [1, 2, 3], we want to have the following intervals:
        (-inf, 1.5), [1.5, 2.5), [2.5, inf).
        This function returns creates the bounds without the infs: [1.5, 2.5].
        If add_inf is True, then it returns [-inf, 1.5, 2.5, inf].
        """

        right_midpoints = self[1:]
        left_midpoints = self[:-1]
        bound = (right_midpoints + left_midpoints) / 2

        if add_inf:
            bound = np.concatenate([[-np.inf], self, [np.inf]])

        return bound.view(np.ndarray)

    
    @classmethod
    def create_feature_representation_from_column(cls, repr_type: str, max_size: int, X: np.ndarray) -> 'FeatureRepresentation':
        
        if repr_type == 'quantile':
            return cls.create_quantile_representation(max_size, X)
        elif repr_type == 'unique':
            return cls.create_unique_rounded_representation(max_size, X)
        elif repr_type == 'uniform':
            return cls.create_uniform_representation(max_size, X)
        else:
            raise ValueError('Feature Representation type not supported')


    @classmethod
    def create_quantile_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':
        
        """
        For a given dataset, gather the quantile information for each feature.
        We have n_buckets+1 values because we include the minimum and
        maximum values of each feature. For example, if n_buckets=4, then we
        have 5 values: [min, q1, q2, q3, max].
        In case of duplicate values, we remove them.
        """
        
        n_buckets = max_size
        quantiles = [i/n_buckets for i in range(n_buckets+1)]

        values = np.quantile(x, q=quantiles, interpolation='midpoint')
        values_unique = np.unique(values)

        return values_unique.view(cls)

    
    @classmethod
    def create_unique_rounded_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the unique values.
        The feature representation is a 1D numpy array of shape (n), where n is the number of unique values.
        In case there are too many unique values, we round the values to a certain number of digits.
        """

        unique_values = unique_values_through_rounding(max_size, x)
        return unique_values.view(cls)
    

    @classmethod
    def create_unique_values(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the unique values.
        The feature representation is a 1D numpy array of shape (n), where n is the number of unique values.
        """

        unique_values = np.unique(x)
        return unique_values.view(cls)
    
    @classmethod
    def create_uniform_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the uniform values.
        The feature representation is a 1D numpy array of shape (n), where n is the specified size.
        """

        max_element = np.max(x)
        min_element = np.min(x)
        size = max_size
        uniform_values = np.linspace(min_element, max_element, size)

        return uniform_values.view(cls)



def unique_values_through_rounding(max_size: int, features: np.ndarray) -> np.ndarray:
    """
    We want to find the number of unique values in a feature.
    In case there are too many unique values, we round the values to a certain number of digits.
    We try to pick a number of digits that results in the highest number of unique values 
    that is less than a certain threshold.
    'digits' is the number of digits we round to, but it is not necessarily an integer.
    """

    unique_values = np.unique(features)
    max_dim = max_size

    if len(unique_values) <= max_dim:
        return unique_values
       
    func = min_dist_max_dim(features, max_dim)
    digits = scipy.optimize.minimize(func, 3, method='Nelder-Mead')['x'][0]
    unique_values = get_unique_values(features, digits)

    return unique_values


def get_unique_values(features, digits):
    rounded = (features // 10**(-digits)) * 10**(-digits)
    unique_values = np.unique(rounded)
    return unique_values


def min_dist_max_dim(features, max_dim):

    def f(digits):
        unique_values = get_unique_values(features, digits)
        return (math.log(len(unique_values)) - math.log(max_dim)) ** 2

    return f




class FeatureRepresentationList(ty.List[FeatureRepresentation]):
    """
    This is a list of feature representations
    """

    def __init__(self):
        super().__init__()


    @classmethod
    def create_representations(cls, repr_type: str, max_size: int, X: np.ndarray):

        num_features = X.shape[1]
        bounds = cls()

        for i_feature in range(num_features):
            bound = FeatureRepresentation.create_feature_representation_from_column(repr_type, max_size, X[:, i_feature])
            bounds.append(bound)

        return bounds
    

    @classmethod
    def create_unique_values(cls, max_size: int, X: np.ndarray):

        num_features = X.shape[1]
        bounds = cls()

        for i_feature in range(num_features):
            bound = FeatureRepresentation.create_unique_values(max_size, X[:, i_feature])
            bounds.append(bound)

        return bounds