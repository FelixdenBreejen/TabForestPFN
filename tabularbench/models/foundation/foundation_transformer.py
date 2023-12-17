import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabularbench.models.foundation.embedding import (
    FoundationEmbeddingYFloat, FoundationEmbeddingYInteger,
    FoundationObservationEmbedding)


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
            linear_attention: bool,
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
        self.linear_attention = linear_attention

        self.x_embedding = nn.Linear(n_features, dim)

        if self.y_as_float_embedding:
            self.y_embedding = FoundationEmbeddingYFloat(dim)
        else:
            self.y_embedding = FoundationEmbeddingYInteger(n_classes, dim)

        self.obs_embedding_support = FoundationObservationEmbedding(dim)
        self.obs_embedding_query__ = FoundationObservationEmbedding(dim)

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):

            if linear_attention:
                attention = LinearAttention(dim, n_heads)
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

            if self.linear_attention:
                module_dict['attention'].init_weights()
            else:
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

        y_query is (batch_size, n_observations_query, n_classes)

        syntax:
        b = batch size
        n = number of observations
        d = dimension of embedding
        c = number of classes
        """

        x_query__ = x_query

        batch_size = x_support.shape[0]
        n_obs_support = x_support.shape[1]
        n_obs_query__ = x_query__.shape[1]

        padding_mask = torch.zeros((batch_size, n_obs_support), dtype=torch.bool, device=x_support.device)
        padding_mask[y_support == -100] = True
        
        x_support = self.x_embedding(x_support)
        x_query__ = self.x_embedding(x_query__)
    
        y_support, y_query__ = self.y_embedding(y_support, n_obs_query__)

        obs_embedding_support = self.obs_embedding_support(batch_size, n_obs_support)
        obs_embedding_query__ = self.obs_embedding_query__(batch_size, n_obs_query__)

        support = x_support + y_support + obs_embedding_support
        query__ = x_query__ + y_query__ + obs_embedding_query__

        x, pack = einops.pack((support, query__), 'b * d')
        
        for module_dict in self.layers:

            x_residual = x
            support, query__ = einops.unpack(x, pack, 'b * d')
            att_support = module_dict['attention'](support, support, support, key_padding_mask=padding_mask, need_weights=False)[0]
            att_query__ = module_dict['attention'](query__, support, support, key_padding_mask=padding_mask, need_weights=False)[0]
            x = einops.pack((att_support, att_query__), 'b * d')[0]
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

        support, query__ = einops.unpack(x, pack, 'b * c')

        return query__



    


class LinearAttention(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            n_heads: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads

        self.Q = nn.Linear(dim, dim)
        self.K = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.O = nn.Linear(dim, dim)


    def init_weights(self):
        nn.init.zeros_(self.O.weight)
        nn.init.zeros_(self.O.bias)


    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            key_padding_mask: torch.Tensor, 
            need_weights: bool
        ) -> torch.Tensor:
        """
        b = batch size
        n = number of samples (dataset size)
        h = heads
        d = dimension of embedding

        query is (b, n, d)
        key is (b, n, d)
        value is (b, n, d)

        attention weights will be (b, h, d, d)
        output will be (b, n, d)
        """

        query = self.Q(query)
        key   = self.K(key  )   
        value = self.V(value)

        key.masked_fill_(key_padding_mask[:, :, None], -1e9)

        query = torch.nn.functional.softmax(query, dim=1)
        key   = torch.nn.functional.softmax(key  , dim=2)

        query = einops.rearrange(query, 'b n (h d) -> b h n d', h=self.n_heads)
        key =   einops.rearrange(key  , 'b n (h d) -> b h n d', h=self.n_heads)
        value = einops.rearrange(value, 'b n (h d) -> b h n d', h=self.n_heads)

        kv     = torch.einsum('b h n d, b h n e -> b h d e', key  , value)       
        output = torch.einsum('b h n d, b h d e -> b h n e', query, kv   )
        output = einops.rearrange(output, 'b h n d -> b n (h d)')

        output = self.O(output)

        return output, None