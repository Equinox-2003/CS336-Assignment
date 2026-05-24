import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1E-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_type = x.dtype
        x = x.to(torch.float32)
        den = (x**2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(den + self.eps)
        # 注意是hadamard乘积
        return (self.g * x).to(x_type)
    