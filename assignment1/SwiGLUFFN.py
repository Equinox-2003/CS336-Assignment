import torch
from torch import nn
from Linear import Linear

# 注意运算都是 Hadamard 乘积
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor):
        w1x = self.W1(x)
        w3x = self.W3(x)
        return self.W2(self._SiLU(w1x) * w3x)

    def _SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)