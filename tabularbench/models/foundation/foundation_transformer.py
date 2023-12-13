import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabularbench.models.foundation.cosformer_attention import CosformerAttention


class FoundationTransformer(nn.Module):

    def __init__(
            self,
            n_features: int,
            n_classes: int, 
            dim: int,
            n_layers: int,
            n_heads: int,
            attn_dropout: float,
            y_as_float_embedding: bool,
            reversed_attention: bool,
            use_pretrained_weights: bool,
            path_to_weights: str,
        ) -> None:
        
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attn_dropout = attn_dropout
        self.y_as_float_embedding = y_as_float_embedding
        self.reversed_attention = reversed_attention

        self.x_embedding = nn.Linear(n_features, dim)

        if self.y_as_float_embedding:
            self.y_embedding = FoundationEmbeddingYFloat(dim)
        else:
            self.y_embedding = FoundationEmbeddingYInteger(n_classes, dim)

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):

            if reversed_attention:
                attention = CosformerAttention(dim, n_heads, )
            else:
                attention = nn.MultiheadAttention(dim, n_heads, dropout=attn_dropout, batch_first=True)

            self.layers.append(nn.ModuleDict({
                'layer_norm1': nn.LayerNorm(dim),
                'attention': attention,
                'layer_norm2': nn.LayerNorm(dim),
                'linear1': nn.Linear(dim, dim*2),
                'linear2': nn.Linear(dim*2, dim),
            }))

        self.final_layer1 = nn.Linear(dim, dim*2)
        self.final_layer2 = nn.Linear(dim*2, n_classes)

        if use_pretrained_weights:
            self.load_state_dict(torch.load(path_to_weights))
        else:
            self.init_weights()


    def init_weights(self):

        for module_dict in self.layers:

            if not self.reversed_attention:
                nn.init.zeros_(module_dict['attention'].out_proj.weight)
                nn.init.zeros_(module_dict['attention'].out_proj.bias)
            nn.init.zeros_(module_dict['linear2'].weight)
            nn.init.zeros_(module_dict['linear2'].bias)
            

    def forward(self, x_support: torch.Tensor, y_support: torch.Tensor, x_query: torch.Tensor):

        """
        x_support is (batch_size, n_observations_support, n_features)
        y_support is (batch_size, n_observations_support)

        x_query is (batch_size, n_observations_query, n_features)

        returns:

        y_query is (batch_size, n_observation_query, n_classes)

        syntax:
        b = batch size
        n = number of observations
        d = dimension of embedding
        c = number of classes
        """

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query = x_query.shape[1]

        padding_mask = torch.zeros((batch_size, n_obs_support), dtype=torch.bool, device=x_support.device)
        padding_mask[y_support == -100] = True
        
        x_support = self.x_embedding(x_support)
        x_query = self.x_embedding(x_query)
    
        y_support, y_query = self.y_embedding(y_support, n_obs_query)

        support = x_support + y_support
        query = x_query + y_query

        x, pack = einops.pack((support, query), 'b * d')
        
        for module_dict in self.layers:

            x_residual = x
            support, query = einops.unpack(x, pack, 'b * d')
            support_att = module_dict['attention'](support, support, support, key_padding_mask=padding_mask, need_weights=False)[0]
            query_att = module_dict['attention'](query, support, support, key_padding_mask=padding_mask, need_weights=False)[0]
            x = einops.pack((support_att, query_att), 'b * d')[0]
            x = x_residual + x
            x = module_dict['layer_norm1'](x)
            x_residual = x
            x = module_dict['linear1'](x)
            x = torch.nn.functional.gelu(x)
            x = module_dict['linear2'](x)
            x = x_residual + x
            x = module_dict['layer_norm2'](x)

        x = self.final_layer1(x)
        x = F.gelu(x)
        x = self.final_layer2(x)

        support, query = einops.unpack(x, pack, 'b * c')

        return query



class FoundationEmbeddingYFloat(torch.nn.Module):

    def __init__(
            self,
            dim: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim

        self.y_embedding = nn.Linear(1, dim)


    def forward(self, y_support: torch.Tensor, n_obs_query: int) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = y_support.shape[0]

        y_support = y_support.type(torch.float32)
        y_support = einops.rearrange(y_support, 'b n -> b n 1')

        y_support = self.y_embedding(y_support)
        y_query = torch.zeros((batch_size, n_obs_query, self.dim), device=y_support.device, dtype=torch.float32)

        return y_support, y_query
    


class FoundationEmbeddingYInteger(torch.nn.Module):

    def __init__(
            self,
            n_classes: int, 
            dim: int,
        ) -> None:
        
        super().__init__()

        self.n_classes = n_classes
        self.dim = dim

        self.y_embedding = nn.Embedding(n_classes, dim)
        self.y_padding = nn.Embedding(1, dim, padding_idx=0) # padding is modeled as a separate class
        self.y_mask = nn.Embedding(1, dim) # masking is also modeled as a separate class


    def forward(self, y_support: torch.Tensor, n_obs_query: int) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = y_support.shape[0]
        n_obs_support = y_support.shape[1]

        # padding is given as -100. We turn the padding into a 'separate class'
        # because padding is ignored in the attention, this should make no difference whatsoever

        y_support_pad = y_support == -100

        y_sup = torch.zeros((batch_size, n_obs_support, self.dim), device=y_support.device, dtype=torch.float32)
        y_sup[ y_support_pad] = self.y_padding(   y_support[ y_support_pad] + 100 )
        y_sup[~y_support_pad] = self.y_embedding( y_support[~y_support_pad]       )

        y_query = torch.zeros((batch_size, n_obs_query), device=y_support.device, dtype=torch.int64)
        y_query = self.y_mask(y_query)

        return y_sup, y_query
    


class ReversedAttention(torch.nn.Module):

    def __init__(
            self,
            dim: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)


    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            key_padding_mask: torch.Tensor, 
            need_weights: bool
        ) -> torch.Tensor:

        batch_size = query.shape[0]

        query = self.Q(query)
        key = self.K(key)
        value = self.V(value)
        
        if key_padding_mask is not None:
            key = key.masked_fill(key_padding_mask[:, :, None], 0)

        # attention_weights = torch.einsum('b n d, b m d -> b n m', query, key) / self.dim
        # attention_weights = torch.softmax(attention_weights, dim=-1)
        # QKtV = torch.einsum('b n m, b m d -> b n d', attention_weights, value)

        attention_weights = torch.einsum('b n d, b n e -> b d e', key, value) / math.sqrt(batch_size)
        attention_weights = torch.softmax(attention_weights, dim=1)
        QKtV = torch.einsum('b n d, b d e -> b n e', query, attention_weights)

        return QKtV, None


