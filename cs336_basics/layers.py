import torch
import math
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device = None, dtype = None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weights = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        stdev = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights.T)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding, mean = 0, std = 1, a = -3, b = 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.to(torch.long)
        return self.embedding[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.square(x).mean(dim = -1, keepdim = True) + self.eps)
        x  = x * rms * self.weights
        return x.to(in_type)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device = None, dtype = None):
        super().__init__()

        if d_ff is None:
            d_ff = (8 / 3) * d_model
            d_ff = 64 * math.ceil(d_ff / 64)

        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        stdev = (2 / (d_ff + d_model)) ** 0.5
        torch.nn.init.trunc_normal_(self.w1_weight, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.w2_weight, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)
        torch.nn.init.trunc_normal_(self.w3_weight, mean = 0, std = stdev, a = -3 * stdev, b = 3 * stdev)

    def forward(self, x: torch.Tensor):
        w1_x = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        result = einsum(self.silu(w1_x) * w3_x, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
        return result
    
    def silu(self, x: torch.Tensor):
        return x * torch.sigmoid(x)




