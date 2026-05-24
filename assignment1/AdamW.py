import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")

        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")

        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad

                    if grad.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    state = self.state[p]

                    if len(state) == 0:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    m = state["m"]
                    v = state["v"]

                    state["t"] += 1
                    t = state["t"]

                    if weight_decay != 0:
                        p.add_(p, alpha=-lr * weight_decay)

                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    lr_t = lr * ((1 - beta2 ** t) ** 0.5) / (1 - beta1 ** t)

                    p.addcdiv_(m, v.sqrt().add(eps), value=-lr_t)

        return loss
