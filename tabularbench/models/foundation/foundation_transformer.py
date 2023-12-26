import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabularbench.models.foundation.embedding import (
    FoundationEmbeddingX, FoundationEmbeddingYFloat,
    FoundationEmbeddingYInteger)


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

        self.x_embedding = FoundationEmbeddingX(dim, n_features)

        if self.y_as_float_embedding:
            self.y_embedding = FoundationEmbeddingYFloat(dim)
        else:
            self.y_embedding = FoundationEmbeddingYInteger(n_classes, dim)

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):

            att = MultiheadAttention(dim, n_heads) 

            self.layers.append(nn.ModuleDict({
                'layer_norm1': nn.LayerNorm(dim),
                'attention_lin': att,
                'attention_mha': att,
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

            module_dict['attention_lin'].init_weights()
            module_dict['attention_mha'].init_weights()
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

        x_support, x_query__ = self.x_embedding(x_support, x_query__)
        y_support, y_query__ = self.y_embedding(y_support, n_obs_query__)

        support = x_support + y_support
        query__ = x_query__ + y_query__

        x, pack = einops.pack((support, query__), 'b * d')
        
        for module_dict in self.layers:

            x_residual = x
            support, query__ = einops.unpack(x, pack, 'b * d')
            att_support = module_dict['attention_lin'](support, support, support, key_padding_mask=padding_mask)
            att_query__ = module_dict['attention_mha'](query__, support, support, key_padding_mask=padding_mask)
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



class MultiheadAttention(torch.nn.Module):

    def __init__(self, dim: int, n_heads: int) -> None:
        
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads

        self.att = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)


    def init_weights(self):
        nn.init.zeros_(self.att.out_proj.weight)
        nn.init.zeros_(self.att.out_proj.bias)

    
    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            key_padding_mask: torch.Tensor
        ) -> torch.Tensor:
        """
        b = batch size
        n = number of samples (dataset size)
        h = heads
        d = dimension of embedding

        query is (b, n, d)
        key is (b, n, d)
        value is (b, n, d)

        attention weights will be (b, h, n, n)
        output will be (b, n, d)
        """

        output, weights = self.att(query, key, value, key_padding_mask=key_padding_mask, need_weights=False)
        return output



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
            key_padding_mask: torch.Tensor
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

        query = 1+torch.nn.functional.elu(query)
        key   = 1+torch.nn.functional.elu(key  )

        key.masked_fill_(key_padding_mask[:, :, None], 0.)

        query = einops.rearrange(query, 'b n (h d) -> b h n d', h=self.n_heads)
        key =   einops.rearrange(key  , 'b n (h d) -> b h n d', h=self.n_heads)
        value = einops.rearrange(value, 'b n (h d) -> b h n d', h=self.n_heads)

        # query = query * (self.dim // self.n_heads) ** -0.5
        # key = key * (self.dim // self.n_heads) ** -0.5

        query = torch.nn.functional.normalize(query, dim=2)
        key = torch.nn.functional.normalize(key, dim=3)

        kv = torch.einsum('b h n d, b h n e -> b h d e', key, value)
        output = torch.einsum('b h n d, b h d e -> b h n e', query, kv)
        output = einops.rearrange(output, 'b h n d -> b n (h d)')

        output = self.O(output)

        return output
    


class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Edited from https://github.com/Amshaker/SwiftFormer/blob/main/models/swiftformer.py
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(
            self,
            dim: int,
            n_heads: int,
        ) -> None:
        
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads

        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)

        self.w_g = nn.Parameter(torch.randn(dim, 1))
        self.scale_factor = dim ** -0.5
        self.Proj = nn.Linear(dim, dim)
        self.final = nn.Linear(dim, dim)


    def init_weights(self):
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)


    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor, 
            key_padding_mask: torch.Tensor,
        ) -> torch.Tensor:

        query = self.to_query(query)
        key = self.to_key(key)

        key.masked_fill_(key_padding_mask[:, :, None], 0.)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1
        G = torch.sum(A * query, dim=1) # BxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out
