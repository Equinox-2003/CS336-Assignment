import torch
from torch import nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    x: [batch_size, seq_len, d_model]
    x.max is a tuple: (values, indices)
    so x.max()[0] is a scalar, or 1x1 tensor
    '''

    ma = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - ma)
    Z_theta = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / Z_theta
