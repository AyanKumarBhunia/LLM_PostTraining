---
title: "Frontier Training Interview (1): Transformers, Architecture, and Long Context"
date: 2026-06-07 12:00:00 +0000
series_order: 2
categories: [interview-prep]
tags: [llm, transformers, attention, rope, long-context, architecture]
pin: false
math: false
permalink: /posts/frontier-training-interview-transformer-architecture/
---

## Goal

This tutorial prepares you for the architecture-heavy part of a frontier-model training interview.

The interviewer is usually not checking whether you memorized definitions. They are checking whether you can reason from first principles:

1. What problem does the component solve?
2. What tradeoff does it introduce?
3. How would you implement it?
4. How would you test whether it helped?

We will walk through transformers, modern LLM blocks, and long-context methods in that order.

## Step 1: Start From Self-Attention

Self-attention works because each token builds a context-dependent representation by looking at other tokens.

For an input sequence:

`x = [x_1, x_2, ..., x_n]`

each token produces three vectors:

- `Q`: what this token is looking for
- `K`: what this token offers as a match
- `V`: the information this token contributes if selected

The core operation is:

```python
# Q, K, V: [B, H, T, D]
# scores: [B, H, T, T]
scores = Q @ K.transpose(-2, -1)
weights = softmax(scores / sqrt(d_head))  # [B, H, T, T]
out = weights @ V                         # [B, H, T, D]
```

The attention map `weights[i, j]` says how much token `i` reads from token `j`.

### Why Scale by `sqrt(d_head)`?

If `Q` and `K` have dimension `d_head`, their dot product grows in variance as the dimension grows. Large logits push softmax into saturation, which makes gradients small and unstable.

Scaling keeps logits in a healthier range:

```python
scores = scores / math.sqrt(d_head)
```

Interview answer:

> We scale attention logits because dot products become larger as head dimension grows. Without scaling, softmax becomes too sharp early in training, causing unstable gradients.

## Step 2: Understand Why Transformers Replaced RNNs

RNNs process tokens sequentially. Token `t` depends on token `t - 1`, so training cannot fully parallelize across sequence length.

Transformers process all tokens in parallel during training. Attention gives every token a direct path to every other token.

The tradeoff:

- RNNs: cheap per step, but sequential and hard to optimize at long range
- Transformers: expensive attention, but parallel, scalable, and easier to train

This is the main reason transformers won for large-scale language modeling.

## Step 3: Multi-Head Attention

One head computes one attention pattern. Multiple heads allow the model to attend to different relationships at the same layer:

- syntax
- coreference
- local context
- retrieval-like copying
- delimiter or structure tracking

Minimal PyTorch-style pseudocode:

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

        # scores: [B, H, T, T]
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        mask = torch.tril(torch.ones(t, t, device=x.device)).bool()  # [T, T]
        scores = scores.masked_fill(~mask, float("-inf"))           # broadcasts to [B, H, T, T]

        weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        y = weights @ v                          # [B, H, T, D]

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)  # [B, T, C]
```

Why not use one large head?

Because one head gives one softmax distribution per token. Multiple heads give multiple independent attention distributions, which increases expressivity and improves optimization.

## Step 4: Causal Masking

In language modeling, token `t` should predict the next token using only previous tokens.

Without causal masking, token `t` could attend to future tokens and leak the answer during training.

```python
mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # [T, T]
scores = scores.masked_fill(~mask, float("-inf"))       # scores: [B, H, T, T]
```

Interview answer:

> Causal masking enforces the autoregressive factorization. It prevents information leakage from future tokens during training.

## Step 5: Positional Information

Self-attention alone is permutation-equivariant. If you shuffle tokens and apply the same shuffle to the output, vanilla attention cannot know the original order.

So transformers need position information.

Common options:

- learned absolute position embeddings
- sinusoidal position embeddings
- RoPE
- ALiBi

### Sinusoidal Embeddings

Sinusoidal embeddings add deterministic position vectors to token embeddings.

```python
def sinusoidal_positions(seq_len, dim):
    pos = torch.arange(seq_len).float()[:, None]   # [T, 1]
    i = torch.arange(0, dim, 2).float()[None, :]   # [1, D/2]
    angle = pos / (10000 ** (i / dim))             # [T, D/2]

    pe = torch.zeros(seq_len, dim)                 # [T, D]
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe
```

They extrapolate better than learned embeddings because positions are generated by a rule, not by a lookup table limited to trained positions.

### RoPE

RoPE rotates query and key vectors as a function of position. The important property is that attention depends on relative position through the dot product between rotated `Q` and `K`.

Conceptual pseudocode:

```python
def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    # cos, sin: broadcastable to [B, H, T, D/2]
    x_even = x[..., 0::2]  # [B, H, T, D/2]
    x_odd = x[..., 1::2]   # [B, H, T, D/2]

    rotated = torch.stack(
        [x_even * cos - x_odd * sin,
         x_even * sin + x_odd * cos],
        dim=-1,
    )  # [B, H, T, D/2, 2]
    return rotated.flatten(-2)
```

Why does RoPE extrapolate better than learned absolute embeddings?

Because it encodes relative offsets through rotations, and the same rotation rule can be evaluated beyond the original context length. It is not perfect at extrapolation, but it degrades more gracefully than learned absolute position tables.

### ALiBi

ALiBi adds a distance-based bias to attention logits:

```python
scores = scores + slope * relative_distance_bias
```

It encourages attention to prefer nearby tokens while still allowing long-range attention. It is simple and does not require learned position embeddings.

## Step 6: Modern LLM Blocks

A modern decoder-only LLM block usually looks like:

```python
class DecoderBlock(nn.Module):
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

This is a PreNorm architecture: normalize before each sublayer.

### PreNorm vs PostNorm

PostNorm:

```python
x = norm(x + sublayer(x))  # x and sublayer(x): [B, T, d_model]
```

PreNorm:

```python
x = x + sublayer(norm(x))  # x and sublayer(norm(x)): [B, T, d_model]
```

PreNorm is usually more stable for deep transformers because the residual stream has a cleaner gradient path.

### RMSNorm vs LayerNorm

LayerNorm subtracts the mean and divides by standard deviation. RMSNorm only divides by root mean square.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., dim]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms  # [..., dim]
```

RMSNorm is cheaper, simple, and works well in large LLMs.

### SwiGLU

SwiGLU is a gated MLP:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, d_model]
        gate = F.silu(self.w1(x))  # [B, T, hidden_dim]
        up = self.w2(x)            # [B, T, hidden_dim]
        return self.w3(gate * up)  # [B, T, d_model]
```

The gate lets the model modulate which features pass through the MLP.

## Step 7: Grouped Query Attention

During decoding, each generated token needs to read all previous keys and values. The KV cache can become the memory bottleneck.

Multi-query attention shares one set of `K,V` across all query heads. Grouped-query attention is a compromise: many query heads share fewer KV heads.

```python
# q: [batch, q_heads, time, d_head]
# k,v: [batch, kv_heads, time, d_head]
repeat = q_heads // kv_heads
k = k.repeat_interleave(repeat, dim=1)  # [batch, q_heads, time, d_head]
v = v.repeat_interleave(repeat, dim=1)  # [batch, q_heads, time, d_head]
```

Why GQA helps inference:

- smaller KV cache
- less memory bandwidth during decode
- similar quality to full MHA when chosen carefully

## Step 8: Long Context

Attention is `O(n^2)` because every token can attend to every other token.

For sequence length `n`, the attention matrix has shape:

`[batch, heads, n, n]`

Doubling context length roughly quadruples attention work and memory.

### FlashAttention

FlashAttention computes exact attention without materializing the full attention matrix in high-bandwidth memory.

The bottleneck it solves is memory IO, not the mathematical complexity.

Conceptual idea:

```python
for q_block in Q_blocks:
    running_max = -inf
    running_sum = 0
    running_out = 0

    for k_block, v_block in KV_blocks:
        scores = q_block @ k_block.T
        update_online_softmax_statistics()
        update_running_output()

    write_output_block()
```

Interview answer:

> FlashAttention is faster because it tiles attention and uses an online softmax, reducing reads and writes to GPU HBM. It computes exact attention but avoids storing the full `n x n` attention matrix.

### Sliding-Window Attention

Each token attends only to a local window:

```python
allowed = abs(i - j) <= window_size
```

This reduces cost but can hurt tasks requiring long-distance retrieval unless combined with global tokens, memory, or recurrence.

### Linear Attention

Linear attention rewrites attention using kernel features:

```python
softmax(Q @ K.T) @ V
```

is approximated by:

```python
phi(Q) @ (phi(K).T @ V)
```

This can reduce complexity to near-linear in sequence length, but often changes model behavior and can underperform full attention on retrieval-heavy tasks.

### Ring Attention

Ring Attention distributes long sequence attention across devices. Each device owns a chunk of the sequence and circulates key/value blocks around a ring.

It is useful when the context is too long for one device but exact or near-exact attention is desired.

## Step 9: Extending an 8k Model to 128k

A practical plan:

1. Decide whether the model must do retrieval, reasoning over long documents, or just tolerate long prompts.
2. Adjust positional encoding with RoPE scaling, interpolation, YaRN, or another method.
3. Continue training on mixtures of short and long sequences.
4. Use efficient attention kernels and memory-saving parallelism.
5. Evaluate with needle-in-a-haystack, multi-needle retrieval, long-document QA, summarization, and long-context reasoning.

What breaks at 1M tokens?

- attention memory and compute
- KV cache size
- positional extrapolation
- training data scarcity for true long-context tasks
- evaluation reliability
- optimization instability from rare long batches

## Interview Checklist

You should be able to answer:

- Why self-attention works
- Why causal masking is required
- Why RoPE is different from absolute embeddings
- Why RMSNorm, PreNorm, SwiGLU, and GQA are common in LLMs
- Why long context is hard
- What FlashAttention actually optimizes
- How you would extend and evaluate context length

If you can explain these without memorized buzzwords, you are ready for most architecture questions.
