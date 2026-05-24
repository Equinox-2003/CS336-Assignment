import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1E-6) -> None:
    grads = [x.grad for x in parameters if x.grad is not None]
    grad_vals = torch.concat([x.flatten() for x in grads])
    l2 = torch.norm(grad_vals, 2)
    if l2 > max_l2_norm:
        fac = max_l2_norm / (l2 + eps)
        for grad in grads:
            # grad = grad * fac
            # grad *= fac           # 这个写法也是对的
            grad.mul_(fac)
