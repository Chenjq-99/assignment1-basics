from .layers import *

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(
            d_model=d_model, eps=0.00001, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(
            d_model=d_model, eps=0.00001, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, self.rope) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x

        