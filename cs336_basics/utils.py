import math
import torch
import numpy as np
import random
import typing
import os
from typing import Iterable, IO, BinaryIO
def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    lr = min_learning_rate \
        + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) \
        * (max_learning_rate - min_learning_rate)
    
    return lr

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    grad = [p.grad for p in parameters if p.grad is not None]
    if len(grad) == 0:
        return
    
    grad_norm = torch.norm(torch.stack([g.norm(2) for g in grad]))

    if grad_norm > max_l2_norm:
        coef = max_l2_norm / (grad_norm + eps)
        for g in grad:
            g.mul_(coef)
    
    return

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    dtype: torch.dtype = torch.long,
    randomize: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    length = x.shape[0]
    start_indices = random.sample(range(0, length - context_length), batch_size)
    if not randomize:
        start_indices = np.arange(0, batch_size)
    batch = np.zeros((batch_size, context_length))
    targets = np.zeros((batch_size, context_length))
    
    for i, p in enumerate(start_indices):
        batch[i, :] = x[p: p + context_length]
        targets[i, :] = x[p + 1: p + 1 + context_length]
    
    return torch.tensor(batch, device = device, dtype = dtype), \
           torch.tensor(targets, device = device, dtype = dtype)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):

    to_save = {"model": model.state_dict(),
               "optimizer": optimizer.state_dict(),
               "iteration": iteration}
    torch.save(to_save, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer):
    
    loaded = torch.load(src)
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    return loaded["iteration"]
