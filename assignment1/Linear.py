import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.rand(out_features, in_features, device=device, dtype=dtype))
        # no bias
        # approximate initializations given in handout
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x @ self.W.T
    