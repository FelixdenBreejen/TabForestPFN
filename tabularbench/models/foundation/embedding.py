import einops
import torch
import torch.nn as nn


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
    

class FoundationObservationEmbedding(torch.nn.Module):

    def __init__(self, dim: int) -> None:
        
        super().__init__()

        self.dim = dim
        self.max_dim = 2**14
        self.embedding = nn.Embedding(self.max_dim, dim)

    
    def forward(self, batch_size: int, n_obs_support: int, n_obs_query: int) -> torch.Tensor:

        assert n_obs_support + n_obs_query <= self.max_dim, f'Number of observations is too large. Max is {self.max_dim}'

        # Take a random embedding from the pool of embeddings 
        weights = torch.ones((batch_size, self.max_dim), dtype=torch.float32, device=self.embedding.weight.device)
        indices = torch.multinomial(weights, num_samples=n_obs_support+n_obs_query, replacement=False)
        x = self.embedding(indices)
        
        x_support = x[:, :n_obs_support]
        x_query = x[:, n_obs_support:n_obs_support+n_obs_query]
        return x_support, x_query