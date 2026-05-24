import torch
from torch import nn
from SwiGLUFFN import SwiGLUFFN
from multihead_self_attention_rope import multihead_self_attention_rope
from RMSNorm import RMSNorm

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, attn_q_proj_weight: torch.Tensor, attn_k_proj_weight: torch.Tensor, attn_v_proj_weight: torch.Tensor, attn_o_proj_weight: torch.Tensor, ln1_weight: torch.Tensor, ln2_weight: torch.Tensor, ffn_w1_weight: torch.Tensor, ffn_w2_weight: torch.Tensor, ffn_w3_weight: torch.Tensor, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.device = device
        self.theta = theta

        self.attn_q_proj_weight = attn_q_proj_weight
        self.attn_k_proj_weight = attn_k_proj_weight
        self.attn_v_proj_weight = attn_v_proj_weight
        self.attn_o_proj_weight = attn_o_proj_weight

        self.rms1 = RMSNorm(d_model, eps=1E-5, device=device)
        self.rms2 = RMSNorm(d_model, eps=1E-5, device=device)
        self.rms1.load_state_dict({"g": ln1_weight})
        self.rms2.load_state_dict({"g": ln2_weight})

        self.ffn = SwiGLUFFN(d_model, d_ff)
        self.ffn.load_state_dict({"W1.W": ffn_w1_weight, "W2.W": ffn_w2_weight, "W3.W": ffn_w3_weight})

        self.MHA = multihead_self_attention_rope(d_model, num_heads, max_seq_len, theta, device)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        # [batch_size, seq_len, ...]
        # shape[1] = seq_len
        token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
        x1 = self.rms1(in_features)
        x1 = self.MHA(x1, self.attn_q_proj_weight, self.attn_k_proj_weight, self.attn_v_proj_weight, self.attn_o_proj_weight, token_positions)

        x1 = in_features + x1

        x2 = self.rms2(x1)
        x2 = self.ffn(x2)
        return x1 + x2
