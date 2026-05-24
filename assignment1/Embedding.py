import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # also row main vector
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        nn.init.trunc_normal_(self.embeddings, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
    