import numpy as np
import torch


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    # starts: [batch_size]
    # starts[:, None]: [batch_size, 1]
    idx = starts[:, None] + np.arange(context_length)

    inputs = torch.from_numpy(x[idx]).long().to(device)
    targets = torch.from_numpy(x[idx + 1]).long().to(device)

    return inputs, targets
