import torch
from .layers import softmax

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Computes the cross entropy loss between the logits and targets.
    """
    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_softmax = logits - torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    loss = -torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss 
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

