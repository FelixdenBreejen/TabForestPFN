import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class SAINTadapted(nn.Module):

    def __init__(
        self, 
        dim=128, 
        n_layers=2, 
        heads=8, 
        n_classes=10, 
        attn_dropout=0., 
        ff_dropout=0.
    ):
        
        super().__init__()

        self.ffn_dropout = ff_dropout

        self.x_embedding = nn.Linear(1, dim)  # Same linear layer for all features
        self.y_embedding = nn.Embedding(n_classes+1, dim)  # +1 for Masked Language Modeling masking

        self.layers = nn.ModuleList([])

        for _ in range(n_layers):

            self.layers.append(nn.ModuleDict({
                'layer_norm1': nn.LayerNorm(dim),
                'attention_features': MultiheadAttention(d=dim, n_heads=heads, dropout=attn_dropout),
                'layer_norm2': nn.LayerNorm(dim),
                'linear1': nn.Linear(dim, dim*2),
                'linear2': nn.Linear(dim, dim),
                'layer_norm3': nn.LayerNorm(dim),
                'attention_observations': MultiheadAttention(d=dim, n_heads=heads, dropout=attn_dropout),
                'layer_norm4': nn.LayerNorm(dim),
                'linear3': nn.Linear(dim, dim*2),
                'linear4': nn.Linear(dim, dim),
            }))

        self.final_layer = nn.Linear(dim, n_classes)


    def forward(self, x, x_size_mask, y, y_size_mask, y_label_mask):
        """
        x is (batch_size, n_observations, n_features)
        y is (batch_size, n_observations)

        x_size_mask is (batch_size, n_observations, n_features)
        y_size_mask is (batch_size, n_observations)
        y_label_mask is (batch_size, n_observations)

        size masks are to make sure we can batch datasets of different numbers of observations and features.
        label masks are the Masked Language Modeling paradigm masks.

        returns:

        x_logits is (batch_size, n_observations, n_classes)
        """

        x = rearrange(x, 'b n f -> b n f 1')
        x = self.x_embedding(x)

        # We treat the label mask as if it is an extra class
        y_masked = y + 1
        y_masked[y_label_mask] = 0
        y_masked = self.y_embedding(y_masked)

        # We add y as if it is the first feature of x
        x = torch.cat([y_masked[:, :, None], x], dim=2)
        mask = torch.cat([y_size_mask[:, :, None], x_size_mask], dim=2)

        mask_attention_features = mask
        mask_attention_observations = rearrange(mask, 'b n f -> b f n')


        for module_dict in self.layers:

            x = module_dict['layer_norm1'](x)
            x_residual = x
            x = module_dict['attention_features'](x, mask_attention_features)
            x = x_residual + x
            x = module_dict['layer_norm2'](x)
            x_residual = x
            x = module_dict['linear1'](x)
            x = GEGLU(x)
            x = F.dropout(x, self.ffn_dropout, self.training)
            x = module_dict['linear2'](x)
            x = x_residual + x
            x = module_dict['layer_norm3'](x)
            x_residual = x
            x = rearrange(x, 'b n f d -> b f n d')
            x = module_dict['attention_observations'](x, mask_attention_observations)
            x = rearrange(x, 'b f n d -> b n f d')
            x = x_residual + x
            x = module_dict['layer_norm4'](x)
            x_residual = x
            x = module_dict['linear3'](x)
            x = GEGLU(x)
            x = F.dropout(x, self.ffn_dropout, self.training)
            x = module_dict['linear4'](x)
            x = x_residual + x

        x = x[:, :, 0]   # We only care about the first feature, which is the placeholder for y
        x = F.gelu(x)
        x = self.final_layer(x)

        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float
    ) -> None:
        
        if n_heads > 1:
            assert d % n_heads == 0
        
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.d = d
        
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout) if dropout else None


    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask=1 means: this token does NOT participate.

        x.shape = (batch_size, n_observations, n_features, d)
        attention is calculated over the n_features dimension
        """
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = rearrange(q, 'b n f (h d) -> b n f h d', h=self.n_heads)
        k = rearrange(k, 'b n f (h d) -> b n f h d', h=self.n_heads)
        v = rearrange(v, 'b n f (h d) -> b n h f d', h=self.n_heads)

        sim = torch.einsum('b n f h d, b n g h d -> b n h f g', q, k) / math.sqrt(self.d_head)

        if mask is not None:
            m = rearrange(mask, 'b n f -> b n 1 1 f')
            sim += m * -1e10

        attention = F.softmax(sim, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)

        x = attention @ v

        x = rearrange(x, 'b n h f d -> b n f (h d)')

        return x
    


def GEGLU(x):
    x, gates = x.chunk(2, dim=-1)
    return x * F.gelu(gates)