import torch
from torch import nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()

        self.theta = theta

        if d_k % 2 > 0:
            raise ValueError("d_k must be an even positive integer")

        self.theta = theta
        self.d_k = d_k  # dimension of query and key
        self.max_seq_len = max_seq_len
        self.device = device

        exps = 1 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        positions = torch.arange(max_seq_len, device=device)
        # outer(a[0..n), b[0..n)) =>c_ij = ai * bj
        vals = torch.outer(positions, exps)
        self.register_buffer("cos_memo", vals.cos(), persistent=False)
        self.register_buffer("sin_memo", vals.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        x_e = x[..., 0::2]
        x_o = x[..., 1::2]

        cos = self.cos_memo[token_positions]
        sin = self.sin_memo[token_positions]

        # [1, max_seq_len, d_k//2] => [max_seq_len, d_k//2]
        # to make use of broadcast
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)    

        # [batch, max_seq_len, d_k//2]
        new_x_e = x_e * cos - x_o * sin
        new_x_o = x_o * cos + x_e * sin

        # [batch, max_seq_len, d_k//2,2]
        res = torch.stack([new_x_e, new_x_o], dim=-1)
    
        # [batch, max_seq_len, d_k], 然后flatten恰好会把奇偶交错排列
        return res.flatten(-2)
    