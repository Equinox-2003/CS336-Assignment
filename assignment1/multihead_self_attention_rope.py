import torch
from torch import nn
from softmax import softmax
from RoPE import RoPE

class multihead_self_attention_rope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device: torch.device | None = None):
        super().__init__()
        self.d_model = d_model  # input dim
        self.num_heads = num_heads  # num of head
        self.head_dim = d_model // num_heads    # dim of each head
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device)
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = self.head_dim # 注意，每个头的维度是 head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1E9)
        attn_weights = softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V)

    def forward(self, x: torch.Tensor, Wq: torch.Tensor, Wk: torch.Tensor, Wv: torch.Tensor, Wo: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, self.d_model = x.shape

        q = x @ Wq.T  # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ Wk.T  # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ Wv.T  # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        # 对角线上方第一条对角往上都是1
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, seq_len]

        attn = self._attention(q, k, v, mask) # [batch_size, num_heads, seq_len, self.head_dim]
        attn = attn.transpose(1, 2) # # [batch_size, seq_len, num_heads, self.head_dim]
        attn = attn.contiguous().view(batch_size, seq_len, self.d_model)
        return attn @ Wo.T
