import torch
from torch import nn
from softmax import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = Q.shape[-1]
    # batch mat mul and batch mat transpose
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1E9)
    attn_weights = softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
