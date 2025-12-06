import torch
import math
import torch.nn as nn
from einops import einsum, rearrange

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    x = x.exp()
    x = x / (x.sum(dim=dim, keepdim=True) + eps)
    return x

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    
    attn_scores = einsum(Q, K, "... len_q d_k, ... len_k d_k -> ... len_q len_k")
    attn_scores = attn_scores / math.sqrt(d_k)

    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, -math.inf)

    attn_scores = softmax(attn_scores, dim=-1)

    output = einsum(attn_scores, V, "... len_q len_k, ... len_k d_v -> ... len_q d_v")

    return output

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        stdev = (2 / (d_in + d_out)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=stdev, a=-3 * stdev, b=3 * stdev)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.T)
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.to(torch.long)
        return self.weight[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.weight
        return x.to(in_type)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = (8 / 3) * d_model
            d_ff = 64 * math.ceil(d_ff / 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.w2(silu(self.w1(x)) * self.w3(x))
        return result

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=torch.float32):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        idx = torch.arange(0, max_seq_len, device=device, dtype=dtype)  # (max_seq_len,)
        denom = theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k)  # (d_k//2,)
        theta_i_k = idx.unsqueeze(1) / denom.unsqueeze(0)  # (max_seq_len, d_k//2)

        cos_cache = torch.cos(theta_i_k)  # (max_seq_len, d_k//2)
        sin_cache = torch.sin(theta_i_k)  # (max_seq_len, d_k//2)

        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        assert d_k % 2 == 0 and d_k == self.d_k, f"d_k mismatch: {d_k} vs {self.d_k}"

        if token_positions is not None:
            cos_values = self.cos_cache[token_positions]
            sin_values = self.sin_cache[token_positions]
        else:
            cos_values = self.cos_cache[:seq_len]
            sin_values = self.sin_cache[:seq_len]

        x_even = x[..., 0::2]  # (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]   # (..., seq_len, d_k//2)
        
        rotated_even = x_even * cos_values - x_odd * sin_values
        rotated_odd = x_even * sin_values + x_odd * cos_values
        
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        rotated = rotated.flatten(-2) 

        return rotated

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module | None = None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, num_heads * self.d_head, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.d_head, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *leading_dims, seq_len, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q, "... seq (num_heads d_head) -> ... num_heads seq d_head", 
                      num_heads=self.num_heads, d_head=self.d_head)
        k = rearrange(k, "... seq (num_heads d_head) -> ... num_heads seq d_head", 
                      num_heads=self.num_heads, d_head=self.d_head)
        v = rearrange(v, "... seq (num_heads d_head) -> ... num_heads seq d_head", 
                      num_heads=self.num_heads, d_head=self.d_head)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

        output = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        output = rearrange(output, "... num_heads seq d_head -> ... seq (num_heads d_head)")

        output = self.output_proj(output)

        return output