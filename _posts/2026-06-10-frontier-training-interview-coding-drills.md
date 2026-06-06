---
title: "Frontier Training Interview (4): Coding Drills in PyTorch"
date: 2026-06-10 12:00:00 +0000
series_order: 5
categories: [interview-prep]
tags: [llm, pytorch, coding-interview, attention, lora, moe, training-loop]
pin: false
math: false
permalink: /posts/frontier-training-interview-coding-drills/
---

## Goal

This tutorial gives implementation drills for frontier-model training interviews.

The goal is not to memorize code. The goal is to become fluent enough that you can write correct, simple versions on a whiteboard or in a shared editor.

For each exercise, practice explaining:

1. tensor shapes
2. numerical stability
3. memory cost
4. what you would test

## Drill 1: Softmax From Scratch

Stable softmax subtracts the maximum logit before exponentiation.

```python
def softmax(x, dim=-1):
    # x: [..., vocab] if dim=-1
    x = x - x.max(dim=dim, keepdim=True).values
    # exp_x: same shape as x
    exp_x = torch.exp(x)
    # output: same shape as x; sums to 1 along dim
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

Tests:

```python
x = torch.randn(4, 10)      # [batch=4, vocab=10]
p = softmax(x, dim=-1)      # [batch=4, vocab=10]
assert torch.allclose(p.sum(dim=-1), torch.ones(4))
assert torch.all(p >= 0)
```

Interview point:

> Always mention numerical stability. Naive `exp(logits)` can overflow.

## Drill 2: Scaled Dot-Product Attention

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: [B, H, T, D]
    # mask: broadcastable to [B, H, T, T], usually [1, 1, T, T]
    d_head = q.size(-1)
    # scores: [B, H, T, T], score from each query token to each key token
    scores = q @ k.transpose(-2, -1)
    scores = scores / math.sqrt(d_head)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # weights: [B, H, T, T], normalized over key positions
    weights = torch.softmax(scores, dim=-1)
    # out: [B, H, T, D]
    out = weights @ v
    return out, weights
```

Shape check:

```text
q:       [B, H, T, D]
k.T:     [B, H, D, T]
scores:  [B, H, T, T]
v:       [B, H, T, D]
out:     [B, H, T, D]
```

## Drill 3: Causal Masking

```python
def causal_mask(seq_len, device):
    # returns [T, T], where row i can attend to columns <= i
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()

mask = causal_mask(T, x.device)       # [T, T]
mask = mask[None, None, :, :]         # [1, 1, T, T], broadcast over B and H
```

Test:

```python
mask = causal_mask(4, "cpu")
expected = torch.tensor([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]).bool()
assert torch.equal(mask, expected)
```

## Drill 4: LayerNorm and RMSNorm

LayerNorm:

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim], normalize over the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # output: [..., dim]
        return self.weight * x + self.bias
```

RMSNorm:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim], normalize by root mean square over dim
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # output: [..., dim]
        return self.weight * x * rms
```

Interview contrast:

- LayerNorm centers and rescales
- RMSNorm only rescales by root mean square
- RMSNorm is cheaper and common in LLMs

## Drill 5: RoPE

Build frequencies:

```python
def rope_cache(seq_len, dim, device, base=10000):
    # dim is d_head and must be even.
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(seq_len, device=device).float()  # [T]
    angles = positions[:, None] * inv_freq[None, :]           # [T, D/2]
    # cos, sin: [T, D/2]
    return torch.cos(angles), torch.sin(angles)
```

Apply rotations:

```python
def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    # cos, sin: [T, D/2]
    cos = cos[None, None, :, :]  # [1, 1, T, D/2]
    sin = sin[None, None, :, :]  # [1, 1, T, D/2]

    x1 = x[..., 0::2]  # [B, H, T, D/2]
    x2 = x[..., 1::2]  # [B, H, T, D/2]

    y1 = x1 * cos - x2 * sin  # [B, H, T, D/2]
    y2 = x1 * sin + x2 * cos  # [B, H, T, D/2]

    # stack -> [B, H, T, D/2, 2], flatten -> [B, H, T, D]
    return torch.stack([y1, y2], dim=-1).flatten(-2)
```

Use it on queries and keys, not values:

```python
cos, sin = rope_cache(T, d_head, x.device)  # each [T, D/2]
q = apply_rope(q, cos, sin)                 # [B, H, T, D]
k = apply_rope(k, cos, sin)                 # [B, H, T, D]
```

## Drill 6: Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C], where C = d_model
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [B, T, C]

        # each becomes [B, H, T, D]
        q = q.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        mask = causal_mask(t, x.device)[None, None, :, :]  # [1, 1, T, T]
        y, _ = scaled_dot_product_attention(q, k, v, mask) # [B, H, T, D]

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)  # [B, T, C]
```

Common bugs:

- forgetting `.contiguous()` before `.view`
- masking with the wrong shape
- applying softmax over the wrong dimension
- forgetting the `sqrt(d_head)` scale

## Drill 7: Grouped Query Attention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads):
        super().__init__()
        assert q_heads % kv_heads == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d_head = d_model // q_heads

        self.q_proj = nn.Linear(d_model, q_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.d_head, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C], C = d_model
        b, t, c = x.shape

        # q: [B, q_heads, T, D]
        q = self.q_proj(x).view(b, t, self.q_heads, self.d_head).transpose(1, 2)
        # k, v: [B, kv_heads, T, D]
        k = self.k_proj(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.d_head).transpose(1, 2)

        repeat = self.q_heads // self.kv_heads
        k = k.repeat_interleave(repeat, dim=1)  # [B, q_heads, T, D]
        v = v.repeat_interleave(repeat, dim=1)  # [B, q_heads, T, D]

        mask = causal_mask(t, x.device)[None, None, :, :]  # [1, 1, T, T]
        y, _ = scaled_dot_product_attention(q, k, v, mask) # [B, q_heads, T, D]

        # [B, q_heads, T, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)  # [B, T, C]
```

Interview point:

> In production you avoid physically repeating K/V when possible. This version is simple for correctness.

## Drill 8: SwiGLU

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, d_model]
        gate = F.silu(self.gate(x))  # [B, T, hidden_dim]
        up = self.up(x)              # [B, T, hidden_dim]
        return self.down(gate * up)  # [B, T, d_model]
```

## Drill 9: Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, hidden_dim)

    def forward(self, x):
        # x: [B, T, d_model]
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x  # [B, T, d_model]
```

Explain why this is PreNorm and why residual connections matter.

## Drill 10: Minimal GPT

```python
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, hidden_dim, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, T]
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device)  # [T]

        # token_emb(input_ids): [B, T, d_model]
        # pos_emb(pos)[None, :, :]: [1, T, d_model], broadcast across B
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)  # [B, T, d_model]

        x = self.norm(x)       # [B, T, d_model]
        return self.lm_head(x) # [B, T, vocab_size]
```

Training loss:

```python
logits = model(input_ids[:, :-1])  # [B, T-1, vocab_size]
targets = input_ids[:, 1:]         # [B, T-1]

# flatten time and batch: [B*(T-1), vocab_size] vs [B*(T-1)]
loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
```

## Drill 11: KV Cache

```python
class KVCache:
    def __init__(self):
        # k, v will store [B, H_kv, cached_T, D]
        self.k = None
        self.v = None

def append_cache(cache, k_new, v_new):
    # k_new, v_new: [B, H_kv, 1, D] for one decode step
    if cache.k is None:
        cache.k = k_new
        cache.v = v_new
    else:
        # concatenate along time/cache dimension
        cache.k = torch.cat([cache.k, k_new], dim=2)
        cache.v = torch.cat([cache.v, v_new], dim=2)
    return cache
```

Attention with cache:

```python
def decode_attention(q_new, k_new, v_new, cache):
    # q_new: [B, H_q, 1, D]
    # k_new, v_new: [B, H_kv, 1, D]
    cache = append_cache(cache, k_new, v_new)
    # cache.k: [B, H_kv, cached_T, D]
    # If using GQA, cache.k/cache.v may be repeated or read by grouped kernels.
    scores = q_new @ cache.k.transpose(-2, -1)
    # scores: [B, H_q, 1, cached_T]
    scores = scores / math.sqrt(q_new.size(-1))
    weights = torch.softmax(scores, dim=-1)  # [B, H_q, 1, cached_T]
    out = weights @ cache.v                  # [B, H_q, 1, D]
    return out, cache
```

Interview point:

> Real implementations preallocate cache tensors instead of concatenating every step.

## Drill 12: LoRA

LoRA freezes the base weight and learns a low-rank update:

```python
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.A = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / rank

    def forward(self, x):
        # x: [..., in_dim]
        base = F.linear(x, self.weight)        # [..., out_dim]
        low_rank = F.linear(x, self.A)         # [..., rank]
        update = F.linear(low_rank, self.B)    # [..., out_dim]
        return base + self.scale * update      # [..., out_dim]
```

Why initialize `B` to zero?

The module initially behaves like the frozen base model, then learns the update.

## Drill 13: Toy MoE Routing

```python
class ToyMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, n_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, hidden_dim)
            for _ in range(n_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: [B, T, C], C = d_model
        router_logits = self.router(x)  # [B, T, n_experts]
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        # weights, indices: [B, T, top_k]
        weights = torch.softmax(weights, dim=-1)  # [B, T, top_k]

        out = torch.zeros_like(x)  # [B, T, C]
        for slot in range(self.top_k):
            expert_idx = indices[..., slot]             # [B, T]
            expert_weight = weights[..., slot][..., None] # [B, T, 1]

            for i, expert in enumerate(self.experts):
                mask = expert_idx == i  # [B, T]
                if mask.any():
                    # x[mask]: [num_tokens_for_expert_i, C]
                    out[mask] += expert_weight[mask] * expert(x[mask])

        return out
```

This is clear but inefficient. Production MoE uses grouped dispatch, capacity limits, and load-balancing losses.

Load-balancing intuition:

```python
router_probs = torch.softmax(router_logits, dim=-1)    # [B, T, n_experts]
tokens_per_expert = router_probs.mean(dim=(0, 1))      # [n_experts]
load_balance_loss = n_experts * (tokens_per_expert.square().sum())
```

## Drill 14: Training Loop With Accumulation and BF16

```python
for step, batch in enumerate(loader):
    optimizer.zero_grad(set_to_none=True)

    for micro in split_batch(batch, grad_accum_steps):
        # micro["input_ids"]: [micro_B, T]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(micro["input_ids"][:, :-1])  # [micro_B, T-1, vocab]
            targets = micro["input_ids"][:, 1:]         # [micro_B, T-1]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),    # [micro_B*(T-1), vocab]
                targets.reshape(-1),                    # [micro_B*(T-1)]
            )
            loss = loss / grad_accum_steps

        loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

Explain:

- why loss is divided by `grad_accum_steps`
- why gradients are clipped before optimizer step
- why `zero_grad` is called once per full batch

## Drill 15: Cosine Scheduler

```python
def cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

## Drill 16: Perplexity Computation

Perplexity is exponentiated average negative log-likelihood.

```python
@torch.no_grad()
def compute_perplexity(model, loader):
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        ids = batch["input_ids"].to(model.device)  # [B, T]
        logits = model(ids[:, :-1])                # [B, T-1, vocab]
        targets = ids[:, 1:]                       # [B, T-1]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),   # [B*(T-1), vocab]
            targets.reshape(-1),                   # [B*(T-1)]
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += targets.numel()

    return math.exp(total_loss / total_tokens)
```

## Drill 17: Chinchilla Compute Estimate

```python
def estimate_training_compute(params, tokens):
    return 6 * params * tokens

def estimate_tokens_for_budget(compute_budget, params):
    return compute_budget / (6 * params)

def estimate_params_for_budget(compute_budget, tokens):
    return compute_budget / (6 * tokens)
```

This is a rough dense-transformer estimate, not an exact profiler result.

## Drill 18: Debug a Diverging Training Loop

Add checks:

```python
if not torch.isfinite(loss):
    print("non-finite loss", step)
    save_debug_batch(batch)
    break

gnorm = grad_norm(model)
if not torch.isfinite(gnorm):
    print("non-finite grad norm", step)
    break
```

Then inspect:

- learning rate
- input IDs and labels
- loss mask
- gradient norm
- activation norm
- mixed precision overflow
- bad data batch
- distributed reduction correctness

## How to Practice

Use this sequence:

1. Implement softmax, attention, masking, RMSNorm.
2. Implement MHA, RoPE, SwiGLU, and a transformer block.
3. Train a tiny GPT on a small text file.
4. Add gradient accumulation and BF16.
5. Add KV cache generation.
6. Add LoRA or toy MoE.
7. Profile memory and speed.
8. Explain every tensor shape out loud.

For interviews, correctness and clarity matter more than cleverness. Write the simplest version first, then discuss how production systems optimize it.
