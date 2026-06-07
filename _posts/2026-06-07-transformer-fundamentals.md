---
title: "Section A: Transformer Fundamentals"
date: 2026-06-07 12:00:00 +0000
series_order: 1
categories: [interview-prep]
tags: [llm, transformers, attention, positional-encoding, rope, alibi]
pin: true
math: false
permalink: /posts/transformer-fundamentals/
---

## Goal

This is the first section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover the core transformer fundamentals:

1. Why self-attention works
2. Why transformers replaced RNNs
3. Why self-attention is different from convolution
4. What `Q`, `K`, and `V` mean
5. Why attention scores are scaled
6. Why multiple heads are useful
7. Why one large head is not equivalent
8. What attention maps represent
9. Why causal masking is needed
10. How cross-attention differs from self-attention
11. Why positional information is needed
12. Sinusoidal positional embeddings
13. RoPE
14. ALiBi
15. Why RoPE extrapolates better than learned embeddings

The goal is not only to memorize answers. The goal is to be able to reason from tensor shapes and model behavior.

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width, also called `d_model`
- `H` = number of attention heads
- `D` = head dimension, usually `C // H`
- `Vocab` = vocabulary size

## 1. Why Does Self-Attention Work?

Self-attention works because every token can build a new representation by reading from other tokens in the same sequence.

Suppose the input is:

```text
The animal did not cross the road because it was tired.
```

To understand what `it` refers to, the model may need to look back at `animal`, not only at nearby words. Self-attention gives each token a direct path to every other token.

At a high level:

```python
# x: [B, T, C]
# Each token starts as a vector of size C.

q = W_q(x)  # [B, T, C], what each token is searching for
k = W_k(x)  # [B, T, C], what each token offers as a match
v = W_v(x)  # [B, T, C], what information each token contributes

scores = q @ k.transpose(-2, -1)  # [B, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, T, T]
out = weights @ v  # [B, T, C]
```

The output token at position `i` becomes a weighted mixture of value vectors from all positions `j`.

So self-attention works because it learns:

- what each token should search for
- which other tokens match that search
- how much information to copy from each matched token

Commentary: Attention is not literally "understanding" by itself. It is a differentiable routing mechanism. The deeper transformer stack, MLPs, residual stream, and training objective make this routing useful.

## 2. Why Did Transformers Replace RNNs?

RNNs process tokens one after another:

```text
h_1 -> h_2 -> h_3 -> ... -> h_T
```

This creates two problems:

- Training is hard to parallelize across time.
- Long-range dependencies must pass through many recurrent steps.

Transformers process all tokens in parallel during training:

```python
# x: [B, T, C]
# All T token vectors are processed together in one matrix/tensor operation.
logits = transformer(x)  # [B, T, Vocab]
```

Self-attention gives token `i` a direct path to token `j`, regardless of distance:

```python
# attention_scores: [B, H, T, T]
# attention_scores[:, :, i, j] connects token i to token j directly.
```

Why transformers won:

- Better parallelism on GPUs/TPUs
- Shorter gradient paths between distant tokens
- Easier scaling to large models and datasets
- More flexible token-to-token interaction

RNNs are not useless, but transformers became dominant because large-scale training rewards parallel computation.

## 3. What Are the Advantages of Self-Attention Over Convolutions?

Convolutions use local windows. A 1D convolution over text might read only nearby tokens:

```python
conv = nn.Conv1d(in_channels=C, out_channels=C, kernel_size=3, padding=1)

# x: [B, T, C]
x_conv = x.transpose(1, 2)  # [B, C, T], Conv1d expects channels before time
y = conv(x_conv)            # [B, C, T]
y = y.transpose(1, 2)       # [B, T, C]
```

With a kernel size of 3, each token initially sees only its immediate neighborhood. To connect distant tokens, a convolutional model needs many layers, dilation, pooling, or larger kernels.

Self-attention can connect any two positions in one layer:

```python
# q, k, v: [B, H, T, D]
scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
y = weights @ v  # [B, H, T, D]
```

Advantages of self-attention:

- Global receptive field in one layer
- Content-dependent routing
- Better for variable-length dependencies
- Strong fit for language, where relevant context may be far away

Advantages of convolutions:

- Cheaper for long sequences
- Strong local inductive bias
- Often efficient in vision/audio settings

Commentary: Self-attention is not always "better" than convolution. It is better for many language-modeling workloads because global, content-dependent interactions matter a lot.

## 4. Explain Q, K, and V From First Principles

`Q`, `K`, and `V` come from the idea of retrieval.

Imagine each token asks a question:

- `Q` = query: "What am I looking for?"
- `K` = key: "What information do I contain?"
- `V` = value: "What content should I pass along if selected?"

For each token pair `(i, j)`:

```python
# q_i: [D], query vector for token i
# k_j: [D], key vector for token j
score_ij = q_i @ k_j  # scalar
```

If the score is high, token `i` reads more from token `j`.

Full tensor version:

```python
x = token_embeddings  # [B, T, C]

q = q_proj(x)  # [B, T, C]
k = k_proj(x)  # [B, T, C]
v = v_proj(x)  # [B, T, C]

# Split into heads.
q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
k = k.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
v = v.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
out = weights @ v  # [B, H, T, D]
```

Important distinction:

- `Q` and `K` decide where to attend.
- `V` carries the information that gets mixed.

## 5. Why Scale Attention Scores by `sqrt(d_head)`?

The raw attention score is a dot product:

```python
score = q @ k
```

If `q` and `k` have dimension `D`, the dot product tends to grow as `D` grows. Large logits make softmax very sharp:

```python
weights = torch.softmax(scores, dim=-1)
```

If softmax saturates, most probability mass goes to one token and gradients can become small or unstable.

So transformers scale by `sqrt(D)`:

```python
# q, k: [B, H, T, D]
scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
scores = scores / math.sqrt(D)    # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
```

Interview answer:

> We divide by `sqrt(d_head)` to keep attention logits at a stable scale as head dimension grows. Without this, softmax can become too peaked, which hurts optimization.

## 6. Why Use Multiple Attention Heads?

One attention head gives each token one attention distribution.

Multiple heads allow the model to learn several different attention patterns in parallel:

- local syntax
- long-range dependency
- delimiter tracking
- copying names or variables
- attending to previous instructions
- attending to document structure

Pseudocode:

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
        # x: [B, T, C]
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        q, k, v = self.qkv(x).chunk(3, dim=-1)  # each [B, T, C]

        q = q.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
        scores = scores / math.sqrt(D)    # [B, H, T, T]

        weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        y = weights @ v                          # [B, H, T, D]

        y = y.transpose(1, 2).contiguous()  # [B, T, H, D]
        y = y.view(B, T, C)                 # [B, T, C]
        return self.out(y)                  # [B, T, C]
```

Multiple heads are useful because each head has its own projections and its own attention map.

## 7. Why Not Use a Single Large Head?

A single large head has a larger vector dimension, but it still produces only one attention distribution per token.

Compare:

```python
# Single large head
weights = torch.softmax(scores, dim=-1)  # [B, 1, T, T]

# Multi-head attention
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
```

With `H` heads, each token can attend in `H` different ways at the same layer.

For example:

- head 1 attends to the previous token
- head 2 attends to the subject of the sentence
- head 3 attends to a delimiter
- head 4 attends to an instruction earlier in the prompt

One large head can store more features in each vector, but it cannot create multiple independent softmax routing patterns in the same way.

Commentary: This is a useful interview-level explanation. In practice, head importance varies: some heads become redundant, some specialize, and some can be pruned with limited loss. But multiple heads are still a strong architectural prior.

## 8. What Does an Attention Map Represent?

An attention map is the matrix of attention probabilities.

For one batch item and one head:

```python
# weights: [B, H, T, T]
attention_map = weights[0, 0]  # [T, T]
```

Interpretation:

```text
attention_map[i, j] = how much token i reads from token j
```

Rows correspond to query positions. Columns correspond to key/value positions.

Example:

```python
# weights: [B, H, T, T]
row_sums = weights.sum(dim=-1)  # [B, H, T]
# Each row should sum to 1 after softmax.
```

Important warning:

An attention map shows routing probabilities, not a complete explanation of model reasoning. The final output also depends on:

- value vectors
- MLP layers
- residual connections
- layer norms
- later attention layers

Commentary: If asked whether attention maps are explanations, a safe answer is: "They are useful diagnostics, but not full causal explanations."

## 9. Why Is Causal Masking Needed?

Decoder-only LLMs are trained to predict the next token.

For a sequence:

```text
x_1, x_2, x_3, ..., x_T
```

token `x_i` should only use tokens up to position `i`, not future tokens.

Without a causal mask, position `i` could attend to `i + 1`, which may contain the answer.

Mask construction:

```python
T = 5
mask = torch.tril(torch.ones(T, T)).bool()  # [T, T]

# mask =
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]
```

Using it in attention:

```python
# scores: [B, H, T, T]
mask = mask[None, None, :, :]  # [1, 1, T, T], broadcast over B and H
scores = scores.masked_fill(~mask, float("-inf"))  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)            # [B, H, T, T]
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

Interview answer:

> Causal masking is needed to preserve the autoregressive training objective. It prevents each position from seeing future tokens and leaking the target.

## 10. How Does Cross-Attention Differ From Self-Attention?

In self-attention, `Q`, `K`, and `V` all come from the same sequence.

```python
# Decoder self-attention
x = decoder_hidden_states  # [B, T_dec, C]

q = q_proj(x)  # [B, T_dec, C]
k = k_proj(x)  # [B, T_dec, C]
v = v_proj(x)  # [B, T_dec, C]
```

In cross-attention, queries come from one sequence, while keys and values come from another.

Example: a decoder attending to encoder outputs.

```python
# decoder_x: [B, T_dec, C]
# encoder_x: [B, T_enc, C]

q = q_proj(decoder_x)  # [B, T_dec, C]
k = k_proj(encoder_x)  # [B, T_enc, C]
v = v_proj(encoder_x)  # [B, T_enc, C]

q = q.view(B, T_dec, H, D).transpose(1, 2)  # [B, H, T_dec, D]
k = k.view(B, T_enc, H, D).transpose(1, 2)  # [B, H, T_enc, D]
v = v.view(B, T_enc, H, D).transpose(1, 2)  # [B, H, T_enc, D]

scores = q @ k.transpose(-2, -1)  # [B, H, T_dec, T_enc]
weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H, T_dec, T_enc]
out = weights @ v  # [B, H, T_dec, D]
```

Key difference:

- Self-attention: sequence attends to itself.
- Cross-attention: one sequence attends to another sequence.

Examples:

- Machine translation: decoder attends to encoded source sentence.
- Vision-language models: text tokens attend to image features.
- Retrieval-augmented generation: generated tokens attend to retrieved chunks, depending on architecture.

## 11. Why Do Transformers Need Positional Information?

Self-attention by itself does not know token order.

If there were no position information, attention would treat the sequence mostly as a set of token vectors. The model could see which tokens exist, but not reliably where they occur.

For example:

```text
dog bites man
man bites dog
```

These contain the same words but have different meanings. Position matters.

In code, token embeddings alone have no built-in order:

```python
input_ids = torch.tensor([[10, 20, 30]])  # [B=1, T=3]
tok = token_embedding(input_ids)          # [1, 3, C]

# Without positional information, position 0 and position 2 are just slots.
# The model needs an additional signal telling it token order.
```

A common absolute-position approach:

```python
positions = torch.arange(T, device=input_ids.device)  # [T]
pos = position_embedding(positions)                   # [T, C]
x = tok + pos[None, :, :]                             # [B, T, C]
```

Transformers need positional information because language is ordered, and attention alone is permutation-equivariant.

## 12. Explain Sinusoidal Positional Embeddings

Sinusoidal positional embeddings use fixed sine and cosine functions of position.

They are not learned. They are generated by a formula.

```python
def sinusoidal_positions(seq_len, dim, device):
    # seq_len: T
    # dim: C, usually d_model
    positions = torch.arange(seq_len, device=device).float()[:, None]  # [T, 1]
    dims = torch.arange(0, dim, 2, device=device).float()[None, :]     # [1, C/2]

    angles = positions / (10000 ** (dims / dim))  # [T, C/2]

    pe = torch.zeros(seq_len, dim, device=device) # [T, C]
    pe[:, 0::2] = torch.sin(angles)               # [T, C/2]
    pe[:, 1::2] = torch.cos(angles)               # [T, C/2]
    return pe                                     # [T, C]
```

Using them:

```python
# token_emb: [B, T, C]
pe = sinusoidal_positions(T, C, token_emb.device)  # [T, C]
x = token_emb + pe[None, :, :]                     # [B, T, C]
```

Intuition:

- Low-frequency dimensions change slowly across positions.
- High-frequency dimensions change quickly.
- The model receives a smooth, multi-scale signal of position.

Why they were useful:

- no learned position table
- can be evaluated at unseen sequence lengths
- simple and deterministic

## 13. Explain RoPE

RoPE stands for Rotary Positional Embedding.

Instead of adding position embeddings to token vectors, RoPE rotates `Q` and `K` vectors by an angle determined by token position.

Important: RoPE is usually applied to queries and keys, not values.

```python
def rope_cache(seq_len, dim, device, base=10000):
    # seq_len: T
    # dim: D, head dimension; must be even
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # inv_freq: [D/2]

    positions = torch.arange(seq_len, device=device).float()  # [T]
    angles = positions[:, None] * inv_freq[None, :]           # [T, D/2]

    return torch.cos(angles), torch.sin(angles)               # each [T, D/2]
```

Applying RoPE:

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

Using RoPE in attention:

```python
# q, k, v: [B, H, T, D]
cos, sin = rope_cache(T, D, q.device)  # each [T, D/2]

q = apply_rope(q, cos, sin)  # [B, H, T, D]
k = apply_rope(k, cos, sin)  # [B, H, T, D]

scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H, T, T]
out = weights @ v  # [B, H, T, D]
```

Why RoPE is powerful:

- encodes position inside attention scores
- naturally represents relative position through rotations
- widely used in modern LLMs such as Llama-style architectures

Commentary: RoPE does not magically solve all long-context problems. It helps, but long-context performance also depends on training data, attention implementation, optimization, and evaluation.

## 14. Explain ALiBi

ALiBi stands for Attention with Linear Biases.

Instead of adding or rotating position embeddings, ALiBi adds a distance-based bias directly to attention scores.

The bias penalizes faraway tokens. A simple causal version:

```python
def alibi_bias(n_heads, seq_len, slopes, device):
    # n_heads: H
    # seq_len: T
    # slopes: [H], one slope per head

    q_pos = torch.arange(seq_len, device=device)[:, None]  # [T, 1]
    k_pos = torch.arange(seq_len, device=device)[None, :]  # [1, T]

    distance = q_pos - k_pos  # [T, T]
    distance = distance.clamp(min=0)  # [T, T], only past distance in causal attention

    slopes = slopes[:, None, None]  # [H, 1, 1]
    bias = -slopes * distance[None, :, :]  # [H, T, T]
    return bias
```

Using it:

```python
# scores: [B, H, T, T]
# bias: [H, T, T]
scores = scores + bias[None, :, :, :]  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
```

Intuition:

- Nearby tokens get less penalty.
- Farther tokens get more negative bias.
- Different heads can use different slopes.

ALiBi is attractive because it is simple and can extrapolate to longer lengths without learned position tables.

## 15. Why Does RoPE Extrapolate Better Than Learned Embeddings?

Learned absolute positional embeddings use a table:

```python
position_embedding = nn.Embedding(max_seq_len, C)

positions = torch.arange(T, device=device)  # [T]
pos = position_embedding(positions)         # [T, C]
```

If the model was trained with `max_seq_len = 4096`, then positions beyond 4096 are not trained. Extending the table is not enough, because the model has not learned how to use those new position vectors.

RoPE is different because positions are generated by a formula:

```python
cos, sin = rope_cache(seq_len=8192, dim=D, device=device)  # each [8192, D/2]
```

The same rule works at longer positions.

Why RoPE extrapolates better:

- no fixed learned table for each absolute position
- position is encoded through rotation
- dot products between rotated queries and keys naturally depend on relative offsets
- the same mathematical rule can be evaluated beyond the training context

However, the careful answer is:

> RoPE extrapolates better than learned absolute embeddings, but not perfectly. At very long lengths, the model can still fail because it was not trained on those lengths, attention becomes expensive, and the frequency structure of RoPE may behave poorly without scaling methods.

Practical long-context work often combines RoPE with:

- position interpolation
- NTK-style RoPE scaling
- YaRN-like scaling
- continued training on long sequences
- long-context evaluation

## Final Interview Checklist

You should now be able to answer all Section A questions:

- Self-attention works by content-based token routing.
- Transformers replaced RNNs because they parallelize better and shorten long-range paths.
- Self-attention has global, content-dependent receptive fields, unlike local convolutions.
- `Q` searches, `K` matches, and `V` carries content.
- Attention scores are scaled by `sqrt(d_head)` for softmax stability.
- Multiple heads allow multiple routing patterns.
- One large head is not equivalent to many heads because it has only one attention map.
- Attention maps are routing probabilities, not full explanations.
- Causal masking prevents future-token leakage.
- Cross-attention uses queries from one sequence and keys/values from another.
- Transformers need position information because attention alone does not encode order.
- Sinusoidal embeddings provide fixed multi-frequency position signals.
- RoPE rotates queries and keys by position.
- ALiBi adds distance-based bias to attention logits.
- RoPE extrapolates better than learned embeddings because it uses a position rule rather than a fixed learned table.
