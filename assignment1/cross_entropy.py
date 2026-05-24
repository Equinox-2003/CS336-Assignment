import torch

'''
-log {e^xi / sum e^xj} 
= -xi - log {sum e^xj}
= -(xi - m) + log{sum e^{xj-m}}
'''

def CrossEntropyLoss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = inputs.max(dim=-1, keepdim=True)[0]
    inputs = inputs - max_logits

    log_sum = torch.log(torch.exp(inputs).sum(dim=-1))
    target_sum = inputs.gather(dim=-1, index=targets.unsqueeze(-1))
    loss = log_sum - target_sum

    return loss.mean()
