import math

def adamw_accounting(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    """
    Compute peak memory usage for AdamW training, broken down into:
    - parameters
    - gradients
    - optimizer state (AdamW first and second moments)
    - activations (for backpropagation)

    Assumes:
    - float32 (4 bytes per element)
    - d_ff = 4 * d_model
    - SwiGLU FFN (3 matrices) and RMSNorm (scale-only, no bias)
    - Independent input/output embeddings (not weight-tied)
    """
    bytes_per_param = 4  # float32
    d_ff = 4 * d_model

    # 1. Parameters
    embedding_params = 2 * vocab_size * d_model  # input + output embeddings

    # Per transformer block
    qkv_params = 3 * d_model * d_model           # Q, K, V projections
    output_proj_params = d_model * d_model       # attention output projection
    # SwiGLU: W1 (D×F), W2 (D×F), W3 (F×D)
    ffn_params = d_model * d_ff + d_model * d_ff + d_ff * d_model
    rmsnorm_params = 2 * d_model                 # two RMSNorm scales per block

    block_params = qkv_params + output_proj_params + ffn_params + rmsnorm_params
    transformer_params = num_layers * block_params
    final_rmsnorm_params = d_model               # final RMSNorm

    total_params = embedding_params + transformer_params + final_rmsnorm_params

    # 2. Gradients (same size as parameters)
    total_gradients = total_params

    # 3. Optimizer state (AdamW: m and v for each parameter)
    optimizer_state = 2 * total_params

    # 4. Activations (for backprop)
    B, L, D, H, F = batch_size, context_length, d_model, num_heads, d_ff

    input_embed_acts = B * L * D
    rmsnorm_acts = B * L * D
    qkv_acts = B * L * 3 * D
    qk_acts = B * H * L * L
    softmax_acts = B * H * L * L
    value_acts = B * L * D
    output_proj_acts = B * L * D
    # SwiGLU activations: W1_out, W2_out (or gated), W3_out
    ffn_acts = B * L * (F + F + D)

    block_acts = (
        rmsnorm_acts + qkv_acts + qk_acts + softmax_acts +
        value_acts + output_proj_acts + ffn_acts
    )

    final_rmsnorm_acts = B * L * D
    logits_acts = B * L * vocab_size

    total_activations = (
        input_embed_acts +
        num_layers * block_acts +
        final_rmsnorm_acts +
        logits_acts
    )

    # Convert to bytes
    memory_parameters = bytes_per_param * total_params
    memory_gradients = bytes_per_param * total_gradients
    memory_optimizer_state = bytes_per_param * optimizer_state
    memory_activations = bytes_per_param * total_activations
    memory_total = memory_parameters + memory_gradients + memory_optimizer_state + memory_activations

    return {
        "param_count": total_params,
        "parameters": memory_parameters,
        "gradients": memory_gradients,
        "optimizer_state": memory_optimizer_state,
        "activations": memory_activations,
        "total": memory_total
    }

def adamw_flops(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    B, L, D, H, N = batch_size, context_length, d_model, num_heads, num_layers
    F = 4 * D  # d_ff
    V = vocab_size

    # Total parameters
    embedding_params = 2 * vocab_size * d_model
    qkv_params = 3 * d_model * d_model
    output_proj_params = d_model * d_model
    ffn_params = d_model * F + d_model * F + F * d_model
    rmsnorm_params = 2 * d_model
    block_params = qkv_params + output_proj_params + ffn_params + rmsnorm_params
    transformer_params = num_layers * block_params
    final_rmsnorm_params = d_model
    total_params = embedding_params + transformer_params + final_rmsnorm_params

    flops_forward = (
        2 * B* L* V * D # embedding
        # Transformer blocks
        + N * (
            + 3 * B * L * D * D * 2    # QKV: 3 matmuls
            + B * H * L * L * (D // H) * 2  # QK^T
            + B * H * L * L * (D // H) * 2  # PV
            + B * L * D * D * 2        # output proj
            + 3 * B * L * D * F * 2    # SwiGLU
        )
        + 2 * B * L * D * V # logits
    )

    # Backward ≈ 2x forward
    flops_backward = 2 * flops_forward

    # Optimizer update
    """
    # Per parameter AdamW update (strict FLOP count):
    # 1. g_sq = g * g                     → 1
    # 2. m = β1 * m + (1-β1) * g          → 3  (2 mul + 1 add)
    # 3. v = β2 * v + (1-β2) * g_sq       → 3  (2 mul + 1 add)
    # 4. denom = sqrt(v) + ε              → 2  (1 sqrt + 1 add)
    # 5. update = m / denom               → 1  (div)
    # 6. update = update + λ * θ          → 2  (1 mul + 1 add)
    # 7. θ = θ - η * update               → 2  (1 mul + 1 sub)
    # Total = 1 + 3 + 3 + 2 + 1 + 2 + 2 = 14 FLOPs
    """
    flops_optimizer = 14 * total_params

    total_flops = flops_forward + flops_backward + flops_optimizer

    return {
        "forward": flops_forward,
        "backward": flops_backward,
        "optimizer": flops_optimizer,
        "total": total_flops
    }    

if __name__ == "__main__":
    model = "gpt2xl"
    vocab_size = 50257

    if model == "gpt2xl":
        d_model = 1600
        num_layers = 48
        d_ff = 6400
        num_heads = 25
    elif model == "gpt2small":
        d_model = 768
        num_layers = 12
        d_ff = d_model * 4
        num_heads = 12
    elif model == "gpt2medium":
        d_model = 1024
        num_layers = 24
        d_ff = d_model * 4
        num_heads = 16
    elif model == "gpt2large":
        d_model = 1280
        num_layers = 36
        d_ff = d_model * 4
        num_heads = 20
    elif model == 'ours':
        d_model = 512
        num_layers = 4
        d_ff = 1344
        num_heads = 16

    seq_len = 1024

    """
    How much peak memory does running AdamW require? 
    Decompose your answer based on thememory usage of the parameters, activations, gradients, and optimizer state. 
    Express your answerin terms of the batch_size and the model hyperparameters (vocab_size, context_length,num_layers, d_model, num_heads). 
    Assume d_ff = 4 ×d_model
    """

    result = adamw_accounting(1, vocab_size, seq_len, num_layers, d_model, num_heads)

    print(f"Parameters: {result['param_count'] / 1e9:.2f} B")
    print(f"Parameters memory: {result['parameters'] / (1024**3):.2f} GB")
    print(f"Gradients memory: {result['gradients'] / (1024**3):.2f} GB")
    print(f"Optimizer state memory: {result['optimizer_state'] / (1024**3):.2f} GB")
    print(f"Activations memory: {result['activations'] / (1024**3):.2f} GB")
    print(f"Total peak memory: {result['total'] / (1024**3):.2f} GB")

    """
    Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends onthe batch_size. 
    What is the maximum batch size you can use and still fit within 80GB memory?
    """
    memory_bs_1 = adamw_accounting(1, vocab_size, seq_len, num_layers, d_model, num_heads)
    # parameters, gradients, optimizer_state don't based on batch size
    b = memory_bs_1['parameters'] + memory_bs_1['gradients'] + memory_bs_1['optimizer_state']
    a = memory_bs_1['activations']
    maximum_batch_size = math.floor((80 * (1024**3) - b) / a)
    print(f"Maximum batch size: {maximum_batch_size}")

    """
    How many FLOPs does running one step of AdamW take?
    """
    flops = adamw_flops(1, vocab_size, seq_len, num_layers, d_model, num_heads)
    print(f"Forward: {flops['forward'] / 1e9:.2f} GFLOPs")
    print(f"Backward: {flops['backward'] / 1e9:.2f} B")
    print(f"Optimizer: {flops['optimizer'] / 1e9:.2f} GFLOPs")
    print(f"Total: {flops['total'] / 1e9:.2f} GFLOPs")

    """
    AnNVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. 
    Assumingyou are able to get 50% MFU, 
    how long would it take to train a GPT-2 XL for 400K steps and abatch size of 1024 on a single A100?
    """
    total_steps = 400_000
    a100_FLOPS = 19.5 * 1e12
    available_FLOPS = a100_FLOPS * 0.5
    model_flops = adamw_flops(1024, vocab_size, seq_len, num_layers, d_model, num_heads)
    training_time = model_flops['total'] / available_FLOPS * total_steps
    print(f"Training time: {training_time:.2f} seconds", f"or {training_time / (60 * 60 * 24):.2f} days")

