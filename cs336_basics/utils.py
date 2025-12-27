import math
import torch
from typing import Iterable
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

