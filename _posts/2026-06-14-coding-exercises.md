---
title: "Coding Exercises: Attention, Decoding, and LLM Building Blocks"
date: 2026-06-14 12:00:00 +0000
series_order: 8
categories: [interview-prep]
tags: [llm, pytorch, coding-interview, attention, rope, lora, moe]
pin: false
math: false
permalink: /posts/coding-exercises/
---

## Goal

This tutorial covers the most likely coding exercises for a frontier-model Training Team interview.

The goal is not to memorize every line. The goal is to be able to write a clean implementation, explain tensor shapes, and discuss correctness, numerical stability, and memory tradeoffs.

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width, also called `d_model`
- `H` = number of attention heads
- `D` = head dimension, usually `C // H`
- `Vocab` = vocabulary size
- `R` = LoRA rank
- `E` = number of experts
- `K` = top-k experts or beam width, depending on context

## Easy 1. Implement Scaled Dot-Product Attention

Scaled dot-product attention computes:

```text
softmax(QK^T / sqrt(D)) V
```

Implementation:

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q: [B, H, T_q, D]
    # k: [B, H, T_k, D]
    # v: [B, H, T_k, D]
    # mask: broadcastable to [B, H, T_q, T_k], True means "allowed"

    d_head = q.size(-1)

    scores = q @ k.transpose(-2, -1)  # [B, H, T_q, T_k]
    scores = scores / math.sqrt(d_head)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # [B, H, T_q, T_k]

    weights = torch.softmax(scores, dim=-1)  # [B, H, T_q, T_k]
    out = weights @ v                        # [B, H, T_q, D]
    return out, weights
```

What to explain:

- Softmax is over key positions.
- The output length follows the query length.
- `V` must have the same time length as `K`.

## Easy 2. Implement Causal Masking

Causal masking prevents token `i` from seeing future tokens `j > i`.

```python
def causal_mask(seq_len, device):
    # Returns [T, T]
    # mask[i, j] = True if token i may attend to token j.
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

T = 5
mask = causal_mask(T, device="cuda")  # [T, T]
mask = mask[None, None, :, :]         # [1, 1, T, T], broadcasts over B and H
```

Using it:

```python
# scores: [B, H, T, T]
scores = scores.masked_fill(~mask, float("-inf"))  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)            # [B, H, T, T]
```

Test:

```python
expected = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]).bool()

assert torch.equal(causal_mask(4, "cpu"), expected)
```

## Easy 3. Implement Softmax From Scratch

Use the numerically stable version.

```python
def softmax(x, dim=-1):
    # x: any shape, e.g. [B, Vocab]
    x = x - x.max(dim=dim, keepdim=True).values  # same shape as x
    exp_x = torch.exp(x)                         # same shape as x
    return exp_x / exp_x.sum(dim=dim, keepdim=True)  # same shape as x
```

Test:

```python
logits = torch.randn(4, 10)  # [B=4, Vocab=10]
probs = softmax(logits, dim=-1)  # [4, 10]

assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-6)
assert torch.all(probs >= 0)
```

Interview point:

> Subtracting the max prevents overflow in `exp`.

## Easy 4. Implement LayerNorm

LayerNorm normalizes the last dimension.

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [C]
        self.bias = nn.Parameter(torch.zeros(dim))   # [C]
        self.eps = eps

    def forward(self, x):
        # x: [..., C], e.g. [B, T, C]
        mean = x.mean(dim=-1, keepdim=True)  # [..., 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [..., 1]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # [..., C]
        return self.weight * x_hat + self.bias  # [..., C]
```

What to explain:

- Normalize per token, not across the batch.
- `weight` and `bias` are learned per feature dimension.

## Easy 5. Implement RMSNorm

RMSNorm rescales by root mean square without subtracting the mean.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # [C]
        self.eps = eps

    def forward(self, x):
        # x: [..., C], e.g. [B, T, C]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [..., 1]
        return self.weight * x * rms  # [..., C]
```

RMSNorm is common in modern LLMs because it is simple, stable, and cheaper than LayerNorm.

## Easy 6. Implement Positional Encoding

Here is sinusoidal positional encoding.

```python
def sinusoidal_positional_encoding(seq_len, dim, device):
    # seq_len: T
    # dim: C
    positions = torch.arange(seq_len, device=device).float()[:, None]  # [T, 1]
    dims = torch.arange(0, dim, 2, device=device).float()[None, :]     # [1, C/2]

    angles = positions / (10000 ** (dims / dim))  # [T, C/2]

    pe = torch.zeros(seq_len, dim, device=device)  # [T, C]
    pe[:, 0::2] = torch.sin(angles)                # [T, C/2]
    pe[:, 1::2] = torch.cos(angles)                # [T, C/2]
    return pe                                      # [T, C]
```

Using it:

```python
# token_emb: [B, T, C]
pe = sinusoidal_positional_encoding(T, C, token_emb.device)  # [T, C]
x = token_emb + pe[None, :, :]  # [B, T, C]
```

## Easy 7. Implement RoPE

RoPE rotates query and key vectors using position-dependent angles.

```python
def rope_cache(seq_len, dim, device, base=10000):
    # seq_len: T
    # dim: D, must be even
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # inv_freq: [D/2]

    positions = torch.arange(seq_len, device=device).float()  # [T]
    angles = positions[:, None] * inv_freq[None, :]           # [T, D/2]

    return torch.cos(angles), torch.sin(angles)  # each [T, D/2]
```

Apply RoPE:

```python
def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    # cos, sin: [T, D/2]
    cos = cos[None, None, :, :]  # [1, 1, T, D/2]
    sin = sin[None, None, :, :]  # [1, 1, T, D/2]

    x_even = x[..., 0::2]  # [B, H, T, D/2]
    x_odd = x[..., 1::2]   # [B, H, T, D/2]

    y_even = x_even * cos - x_odd * sin  # [B, H, T, D/2]
    y_odd = x_even * sin + x_odd * cos   # [B, H, T, D/2]

    y = torch.stack([y_even, y_odd], dim=-1)  # [B, H, T, D/2, 2]
    return y.flatten(-2)                      # [B, H, T, D]
```

Use RoPE on `q` and `k`, not usually on `v`.

```python
cos, sin = rope_cache(T, D, q.device)  # each [T, D/2]
q = apply_rope(q, cos, sin)            # [B, H, T, D]
k = apply_rope(k, cos, sin)            # [B, H, T, D]
```

## Easy 8. Implement Top-k Sampling

Top-k sampling keeps only the `k` highest-logit tokens.

```python
def top_k_sample(logits, k, temperature=1.0):
    # logits: [B, Vocab]
    logits = logits / temperature  # [B, Vocab]

    values, indices = torch.topk(logits, k, dim=-1)  # each [B, k]
    probs = torch.softmax(values, dim=-1)            # [B, k]

    sampled_local = torch.multinomial(probs, num_samples=1)  # [B, 1]
    sampled_token = indices.gather(dim=-1, index=sampled_local)  # [B, 1]
    return sampled_token
```

What to explain:

- Sampling happens over the top-k subset.
- `gather` maps local top-k indices back to vocabulary IDs.

## Easy 9. Implement Nucleus Sampling

Nucleus sampling, or top-p sampling, keeps the smallest set of tokens whose cumulative probability exceeds `p`.

```python
def nucleus_sample(logits, p=0.9, temperature=1.0):
    # logits: [B, Vocab]
    logits = logits / temperature  # [B, Vocab]
    probs = torch.softmax(logits, dim=-1)  # [B, Vocab]

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    # sorted_probs, sorted_idx: [B, Vocab]

    cumulative = torch.cumsum(sorted_probs, dim=-1)  # [B, Vocab]
    keep = cumulative <= p                           # [B, Vocab]
    keep[..., 0] = True  # always keep at least one token

    filtered_probs = sorted_probs.masked_fill(~keep, 0.0)  # [B, Vocab]
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    sampled_sorted = torch.multinomial(filtered_probs, num_samples=1)  # [B, 1]
    sampled_token = sorted_idx.gather(dim=-1, index=sampled_sorted)    # [B, 1]
    return sampled_token
```

Commentary: Some implementations shift the mask so the first token that pushes cumulative probability above `p` is included. The core idea is the same: sample from a dynamic probability mass.

## Easy 10. Implement Beam Search

Beam search keeps the top `K` partial sequences.

Simplified batch size 1 implementation:

```python
@torch.no_grad()
def beam_search(model, prompt, beam_size=4, max_new_tokens=32, eos_token_id=None):
    # prompt: [1, T_prompt]
    beams = [(prompt, 0.0)]  # list of (tokens [1, T_current], logprob scalar)

    for _ in range(max_new_tokens):
        candidates = []

        for tokens, score in beams:
            # tokens: [1, T_current]
            logits = model(tokens)[:, -1, :]  # [1, Vocab]
            log_probs = torch.log_softmax(logits, dim=-1)  # [1, Vocab]

            top_log_probs, top_ids = torch.topk(log_probs, beam_size, dim=-1)
            # top_log_probs, top_ids: [1, beam_size]

            for j in range(beam_size):
                next_id = top_ids[:, j:j+1]  # [1, 1]
                next_score = score + top_log_probs[0, j].item()
                next_tokens = torch.cat([tokens, next_id], dim=-1)  # [1, T_current + 1]
                candidates.append((next_tokens, next_score))

        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        if eos_token_id is not None:
            if all(seq[0, -1].item() == eos_token_id for seq, _ in beams):
                break

    return beams[0][0]  # [1, T_prompt + generated]
```

What to explain:

- Beam search maximizes sequence log probability approximately.
- It is not the same as sampling.
- It can produce less diverse outputs.

## Medium 1. Implement Multi-Head Attention in PyTorch

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [B, T, C]

        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        mask = None
        if self.causal:
            mask = causal_mask(T, x.device)[None, None, :, :]  # [1, 1, T, T]

        y, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        # y: [B, H, T, D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out(y), weights  # [B, T, C], [B, H, T, T]
```

Common bugs:

- Softmax over the wrong dimension
- Missing causal mask
- Wrong reshape order
- Forgetting `.contiguous()` before `.view`

## Medium 2. Implement FlashAttention Simplified

FlashAttention avoids materializing the full attention matrix in memory. A real implementation uses custom CUDA/Triton kernels. This is a simplified educational version showing blockwise attention and online softmax.

```python
def flash_attention_simplified(q, k, v, block_size=128):
    # q, k, v: [B, H, T, D]
    # Returns: [B, H, T, D]
    B, H, T, D = q.shape

    out = torch.zeros_like(q)  # [B, H, T, D]

    for start_q in range(0, T, block_size):
        end_q = min(start_q + block_size, T)
        q_block = q[:, :, start_q:end_q, :]  # [B, H, Tq, D]
        Tq = q_block.size(2)

        # Online softmax state for each query row.
        m = torch.full((B, H, Tq, 1), -float("inf"), device=q.device)  # [B, H, Tq, 1]
        l = torch.zeros((B, H, Tq, 1), device=q.device)                # [B, H, Tq, 1]
        acc = torch.zeros((B, H, Tq, D), device=q.device)              # [B, H, Tq, D]

        for start_k in range(0, T, block_size):
            end_k = min(start_k + block_size, T)
            k_block = k[:, :, start_k:end_k, :]  # [B, H, Tk, D]
            v_block = v[:, :, start_k:end_k, :]  # [B, H, Tk, D]

            scores = q_block @ k_block.transpose(-2, -1)  # [B, H, Tq, Tk]
            scores = scores / math.sqrt(D)

            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)  # [B, H, Tq, 1]
            exp_old = torch.exp(m - m_new)                                     # [B, H, Tq, 1]
            exp_scores = torch.exp(scores - m_new)                             # [B, H, Tq, Tk]

            l_new = l * exp_old + exp_scores.sum(dim=-1, keepdim=True)         # [B, H, Tq, 1]
            acc = acc * exp_old + exp_scores @ v_block                         # [B, H, Tq, D]

            m, l = m_new, l_new

        out[:, :, start_q:end_q, :] = acc / l  # [B, H, Tq, D]

    return out
```

Commentary: This is not production FlashAttention. It still uses normal PyTorch ops and will not match optimized kernels. The purpose is to demonstrate tiling and online softmax.

## Medium 3. Implement Grouped-Query Attention

GQA has more query heads than key/value heads.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads):
        super().__init__()
        assert q_heads % kv_heads == 0
        assert d_model % q_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_head = d_model // q_heads

        self.q_proj = nn.Linear(d_model, q_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        D = self.d_head

        q = self.q_proj(x).view(B, T, self.q_heads, D).transpose(1, 2)
        # q: [B, H_q, T, D]

        k = self.k_proj(x).view(B, T, self.kv_heads, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, D).transpose(1, 2)
        # k, v: [B, H_kv, T, D]

        repeat = self.q_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)  # [B, H_q, T, D]
        v = v.repeat_interleave(repeat, dim=1)  # [B, H_q, T, D]

        mask = causal_mask(T, x.device)[None, None, :, :]  # [1, 1, T, T]
        y, _ = scaled_dot_product_attention(q, k, v, mask=mask)  # [B, H_q, T, D]

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        return self.out(y)  # [B, T, C]
```

Commentary: In production, kernels avoid physically repeating `k` and `v`.

## Medium 4. Implement KV Caching

KV caching stores past keys and values during autoregressive decoding.

```python
class KVCache:
    def __init__(self, num_layers):
        self.k = [None for _ in range(num_layers)]
        self.v = [None for _ in range(num_layers)]

def append_kv(cache, layer_idx, k_new, v_new):
    # k_new, v_new: [B, H_kv, 1, D]
    if cache.k[layer_idx] is None:
        cache.k[layer_idx] = k_new
        cache.v[layer_idx] = v_new
    else:
        cache.k[layer_idx] = torch.cat([cache.k[layer_idx], k_new], dim=2)
        cache.v[layer_idx] = torch.cat([cache.v[layer_idx], v_new], dim=2)
        # cache.k/v[layer_idx]: [B, H_kv, T_cache + 1, D]

    return cache.k[layer_idx], cache.v[layer_idx]
```

One decode-step attention:

```python
def decode_attention(q_new, k_new, v_new, cache, layer_idx):
    # q_new: [B, H_q, 1, D]
    # k_new, v_new: [B, H_kv, 1, D]
    k_cache, v_cache = append_kv(cache, layer_idx, k_new, v_new)
    # k_cache, v_cache: [B, H_kv, T_cache, D]

    k_attn = repeat_kv_if_needed(k_cache)  # [B, H_q, T_cache, D]
    v_attn = repeat_kv_if_needed(v_cache)  # [B, H_q, T_cache, D]

    scores = q_new @ k_attn.transpose(-2, -1)  # [B, H_q, 1, T_cache]
    weights = torch.softmax(scores / math.sqrt(q_new.size(-1)), dim=-1)  # [B, H_q, 1, T_cache]
    out = weights @ v_attn  # [B, H_q, 1, D]
    return out
```

Commentary: Real systems preallocate paged KV cache blocks instead of repeatedly concatenating tensors.

## Medium 5. Implement Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        # x: [B, T, C]
        attn_out, _ = self.attn(self.attn_norm(x))  # [B, T, C]
        x = x + attn_out                            # [B, T, C]

        mlp_out = self.mlp(self.mlp_norm(x))        # [B, T, C]
        x = x + mlp_out                             # [B, T, C]
        return x
```

This is a PreNorm decoder block.

## Medium 6. Implement GPT Decoder Layer

A GPT decoder layer is a causal transformer block used inside a decoder-only language model.

```python
class GPTDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim):
        super().__init__()
        self.block = TransformerBlock(d_model, n_heads, hidden_dim)

    def forward(self, x):
        # x: [B, T, C]
        return self.block(x)  # [B, T, C]
```

Minimal GPT model:

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, hidden_dim, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)  # [T]

        x = self.token_emb(input_ids)  # [B, T, C]
        x = x + self.pos_emb(pos)[None, :, :]  # [B, T, C]

        for layer in self.layers:
            x = layer(x)  # [B, T, C]

        x = self.norm(x)        # [B, T, C]
        return self.lm_head(x)  # [B, T, Vocab]
```

Training loss:

```python
# input_ids: [B, T]
logits = model(input_ids[:, :-1])  # [B, T-1, Vocab]
targets = input_ids[:, 1:]         # [B, T-1]

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),  # [B*(T-1), Vocab]
    targets.reshape(-1),                  # [B*(T-1)]
)
```

## Medium 7. Implement SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        gate = F.silu(self.gate(x))  # [B, T, hidden_dim]
        up = self.up(x)              # [B, T, hidden_dim]
        h = gate * up                # [B, T, hidden_dim]
        return self.down(h)          # [B, T, C]
```

What to explain:

- `silu(gate)` controls which features pass through.
- Multiplication gives a learned gating mechanism.

## Medium 8. Implement Rotary Embeddings

This is the same RoPE implementation, packaged as a module.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, q, k):
        # q, k: [B, H, T, D]
        T = q.size(2)
        cos, sin = rope_cache(T, self.dim, q.device, base=self.base)
        # cos, sin: [T, D/2]

        q = apply_rope(q, cos, sin)  # [B, H, T, D]
        k = apply_rope(k, cos, sin)  # [B, H, T, D]
        return q, k
```

Use inside attention:

```python
q, k = rotary(q, k)  # each [B, H, T, D]
```

Commentary: Some implementations apply RoPE only to part of the head dimension. If asked, mention that full-dimension RoPE is the simple version.

## Medium 9. Implement LoRA

LoRA freezes a base weight and learns a low-rank update.

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)  # [R, in_dim]
        self.B = nn.Parameter(torch.zeros(out_dim, rank))        # [out_dim, R]
        self.scale = alpha / rank

    def forward(self, x):
        # x: [..., in_dim], e.g. [B, T, in_dim]
        base = F.linear(x, self.weight)      # [..., out_dim]
        low_rank = F.linear(x, self.A)       # [..., R]
        update = F.linear(low_rank, self.B)  # [..., out_dim]
        return base + self.scale * update    # [..., out_dim]
```

Why initialize `B` to zero?

```text
At step 0, update = 0, so the LoRA layer initially matches the frozen base layer.
```

## Medium 10. Implement Mixture-of-Experts Routing

MoE routing sends each token to one or more expert MLPs.

```python
class ToyMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, n_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, hidden_dim)
            for _ in range(n_experts)
        ])
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x):
        # x: [B, T, C]
        router_logits = self.router(x)  # [B, T, E]
        top_logits, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
        # top_logits, top_indices: [B, T, K]

        top_weights = torch.softmax(top_logits, dim=-1)  # [B, T, K]
        out = torch.zeros_like(x)                        # [B, T, C]

        for slot in range(self.top_k):
            expert_ids = top_indices[..., slot]             # [B, T]
            expert_weight = top_weights[..., slot][..., None]  # [B, T, 1]

            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id  # [B, T]
                if mask.any():
                    expert_in = x[mask]  # [tokens_for_expert, C]
                    expert_out = expert(expert_in)  # [tokens_for_expert, C]
                    out[mask] += expert_weight[mask] * expert_out

        return out  # [B, T, C]
```

Simple load-balancing helper:

```python
def load_balancing_loss(router_logits):
    # router_logits: [B, T, E]
    probs = torch.softmax(router_logits, dim=-1)  # [B, T, E]
    prob_per_expert = probs.mean(dim=(0, 1))      # [E]
    return probs.size(-1) * (prob_per_expert.square().sum())  # scalar
```

Commentary: This toy MoE is easy to understand but inefficient. Real MoE uses token dispatch, capacity factors, expert parallelism, grouped matmuls, and carefully designed load-balancing losses.

## Final Coding Checklist

You should be able to implement and explain:

- scaled dot-product attention
- causal masking
- stable softmax
- LayerNorm and RMSNorm
- sinusoidal position encoding
- RoPE and rotary embedding modules
- top-k sampling
- nucleus sampling
- beam search
- multi-head attention
- simplified FlashAttention idea
- grouped-query attention
- KV caching
- transformer block
- GPT decoder layer
- SwiGLU
- LoRA
- MoE routing

For each implementation, practice saying the tensor shapes out loud. In interviews, shape fluency is often the fastest way to show you understand the system.
