---
title: "Section C: Long Context and Efficient Attention"
date: 2026-06-09 12:00:00 +0000
series_order: 3
categories: [interview-prep]
tags: [llm, long-context, flashattention, sparse-attention, linear-attention, rope, yarn]
pin: false
math: false
permalink: /posts/long-context-efficient-attention/
---

## Goal

This is the third section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover long context and efficient attention:

1. Why attention is `O(n^2)`
2. Why context extension becomes difficult
3. FlashAttention
4. Why FlashAttention is faster
5. The bottleneck FlashAttention solves
6. Sliding-window attention
7. Sparse attention
8. Linear attention
9. Pros and cons of linear attention
10. Ring Attention
11. Position interpolation
12. YaRN
13. Extending an 8k model to 128k
14. What breaks at 1M tokens
15. Long-context evaluation

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `H` = number of attention heads
- `D` = head dimension
- `C` = model width, usually `H * D`
- `W` = sliding-window size
- `Vocab` = vocabulary size

## 31. Why Is Attention `O(n^2)`?

Self-attention compares every query token with every key token.

If the sequence length is `T`, then each of `T` query positions attends to `T` key positions. That creates a `T x T` attention matrix.

```python
# q, k, v: [B, H, T, D]
scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H, T, T]
out = weights @ v  # [B, H, T, D]
```

The expensive part is the attention score matrix:

```text
scores shape = [B, H, T, T]
```

So memory and compute scale roughly with:

```text
B * H * T * T
```

If `T` doubles, `T^2` becomes four times larger.

Example:

```text
8k context  -> attention matrix has about 64M positions per head
16k context -> attention matrix has about 256M positions per head
128k context -> attention matrix has about 16B positions per head
```

This is why naive full attention becomes extremely expensive for long contexts.

## 32. Why Does Context Extension Become Difficult?

Extending context length is not just changing `max_seq_len`.

Several things become harder:

1. **Attention cost** grows quadratically with sequence length.
2. **KV cache** grows linearly with generated context length during inference.
3. **Position embeddings** may not extrapolate beyond trained lengths.
4. **Training data** may not contain enough true long-context examples.
5. **Optimization** becomes harder because long sequences use more memory and fewer examples per batch.
6. **Evaluation** becomes harder because retrieval is not the same as reasoning.

Naively increasing context:

```python
# Old training setup
max_seq_len = 8192

# Naive change
max_seq_len = 131072
```

This is usually not enough.

Problems:

```python
# q, k, v: [B, H, T, D]
# At T = 131072, scores would be enormous if materialized.
scores = q @ k.transpose(-2, -1)  # [B, H, 131072, 131072]
```

Even if memory-efficient kernels avoid materializing the full matrix, the model may still fail because its positional encoding and training distribution were not designed for such lengths.

Interview answer:

> Context extension is difficult because it stresses compute, memory, positional encoding, data, optimization, and evaluation at the same time.

## 33. Explain FlashAttention

FlashAttention is an exact attention algorithm that computes the same result as standard attention but avoids materializing the full attention matrix in high-bandwidth GPU memory.

Standard attention conceptually does this:

```python
# q, k, v: [B, H, T, D]
scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
out = weights @ v  # [B, H, T, D]
```

The problem is that `scores` and `weights` are huge.

FlashAttention instead processes queries, keys, and values in blocks:

```python
# Conceptual pseudocode, not production code.
for q_block in split_blocks(q):
    # q_block: [B, H, T_q_block, D]
    running_max = None      # [B, H, T_q_block]
    running_sum = None      # [B, H, T_q_block]
    running_out = 0         # [B, H, T_q_block, D]

    for k_block, v_block in split_blocks(k, v):
        # k_block, v_block: [B, H, T_k_block, D]
        scores_block = q_block @ k_block.transpose(-2, -1)
        # scores_block: [B, H, T_q_block, T_k_block]

        # Use online softmax update instead of storing all scores.
        running_max, running_sum, running_out = online_softmax_update(
            scores_block,
            v_block,
            running_max,
            running_sum,
            running_out,
        )

    write_output(running_out)  # [B, H, T_q_block, D]
```

Key idea:

- Tile the computation.
- Keep small blocks in fast SRAM/shared memory.
- Use an online softmax so the full `T x T` matrix is never stored.

FlashAttention is exact attention, not an approximation.

## 34. Why Is FlashAttention Faster?

FlashAttention is faster because it reduces memory traffic.

GPUs are very fast at matrix multiplication, but reading and writing huge tensors to high-bandwidth memory (HBM) can dominate runtime.

Naive attention writes and reads large intermediate tensors:

```python
scores = q @ k.transpose(-2, -1)  # writes [B, H, T, T] to memory
weights = torch.softmax(scores)   # reads/writes [B, H, T, T]
out = weights @ v                 # reads [B, H, T, T]
```

FlashAttention avoids storing `scores` and `weights` globally:

```python
# Blocks stay close to the compute unit.
# q_block: [B, H, T_q_block, D]
# k_block: [B, H, T_k_block, D]
# v_block: [B, H, T_k_block, D]
```

Why this helps:

- fewer HBM reads/writes
- better cache/SRAM use
- less memory allocation
- often enables larger batch or sequence length

Interview answer:

> FlashAttention is faster because attention is often memory-IO-bound. It computes exact attention with tiling and online softmax, avoiding the expensive materialization of the full attention matrix.

## 35. What Bottleneck Does FlashAttention Solve?

FlashAttention mainly solves the **memory IO bottleneck**, not the theoretical `O(T^2)` compute.

The mathematical work still involves many query-key comparisons:

```text
full attention compute still scales roughly with T^2
```

But the memory behavior improves dramatically because the full attention matrix is not written to and read from HBM.

In simple terms:

```text
Naive attention bottleneck:
  store scores [B, H, T, T]
  store weights [B, H, T, T]
  read them back

FlashAttention:
  compute block
  update online softmax
  keep only output/statistics
```

Important distinction:

- FlashAttention improves efficiency of exact full attention.
- It does not turn full attention into linear attention.
- For extremely long contexts, `T^2` compute can still be too large.

Commentary: If an interviewer asks whether FlashAttention changes the complexity class, the careful answer is no. It changes the practical memory traffic and constant factors, not the full-attention asymptotic compute.

## 36. Explain Sliding-Window Attention

Sliding-window attention restricts each token to attend only to nearby tokens.

Instead of attending to all previous positions:

```text
token i attends to tokens 0 ... i
```

it attends only to a local window:

```text
token i attends to tokens max(0, i-W) ... i
```

Mask construction:

```python
def sliding_window_mask(T, W, device):
    # T: sequence length
    # W: window size
    q_pos = torch.arange(T, device=device)[:, None]  # [T, 1]
    k_pos = torch.arange(T, device=device)[None, :]  # [1, T]

    causal = k_pos <= q_pos          # [T, T]
    within_window = k_pos >= q_pos - W  # [T, T]
    return causal & within_window    # [T, T]
```

Using it:

```python
# scores: [B, H, T, T]
mask = sliding_window_mask(T, W, scores.device)  # [T, T]
scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))  # [B, H, T, T]
weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
```

Effective complexity becomes closer to:

```text
O(T * W)
```

instead of:

```text
O(T^2)
```

Tradeoff:

- much cheaper for long sequences
- good for local dependencies
- weak for tasks requiring information far outside the window

## 37. Explain Sparse Attention

Sparse attention means only some query-key pairs are allowed.

Sliding-window attention is one type of sparse attention, but sparse patterns can be more general.

Examples:

- local window attention
- block sparse attention
- global tokens
- dilated attention
- random sparse attention
- retrieval-selected attention

Block-sparse example:

```python
# x is split into blocks.
# q_blocks: list of [B, H, block_T, D]
# k_blocks: list of [B, H, block_T, D]

allowed_blocks = {
    0: [0],
    1: [0, 1],
    2: [1, 2],
    3: [0, 2, 3],  # maybe a global/retrieval block
}

for q_block_id, q_block in enumerate(q_blocks):
    # q_block: [B, H, block_T, D]
    selected_k = [k_blocks[j] for j in allowed_blocks[q_block_id]]
    selected_v = [v_blocks[j] for j in allowed_blocks[q_block_id]]

    k_cat = torch.cat(selected_k, dim=2)  # [B, H, selected_T, D]
    v_cat = torch.cat(selected_v, dim=2)  # [B, H, selected_T, D]

    scores = q_block @ k_cat.transpose(-2, -1)  # [B, H, block_T, selected_T]
    weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H, block_T, selected_T]
    out_block = weights @ v_cat  # [B, H, block_T, D]
```

Sparse attention reduces compute by not attending everywhere.

Main challenge:

> You must choose a sparse pattern that preserves the information the task needs.

If the correct answer depends on a token that is not reachable by the sparse pattern, the model may fail.

## 38. Explain Linear Attention

Standard attention:

```python
out = softmax(Q @ K.T) @ V
```

requires the `T x T` attention matrix.

Linear attention tries to rewrite attention so it avoids explicitly forming `Q @ K.T`.

One common idea uses a feature map `phi`:

```python
# q, k, v: [B, H, T, D]
q_phi = phi(q)  # [B, H, T, R]
k_phi = phi(k)  # [B, H, T, R]
```

Then approximate attention like:

```python
# First aggregate keys and values.
kv = k_phi.transpose(-2, -1) @ v  # [B, H, R, D]

# Then apply queries.
out = q_phi @ kv  # [B, H, T, D]
```

With proper normalization:

```python
# k_sum: [B, H, R]
k_sum = k_phi.sum(dim=2)

# denom: [B, H, T, 1]
denom = (q_phi * k_sum[:, :, None, :]).sum(dim=-1, keepdim=True)

out = out / (denom + 1e-6)  # [B, H, T, D]
```

The key benefit is that the computation can be closer to:

```text
O(T * R * D)
```

instead of:

```text
O(T^2 * D)
```

where `R` is the feature dimension of the kernel approximation.

Commentary: This is a simplified explanation. Different linear attention methods use different kernels, recurrence forms, normalizations, and stability tricks.

## 39. Pros and Cons of Linear Attention

Pros:

- Better asymptotic scaling for long sequences
- Can support streaming or recurrent-style updates
- Lower memory than full attention
- Useful when exact token-to-token attention is too expensive

Cons:

- Usually approximates or changes softmax attention
- Can underperform on retrieval-heavy tasks
- Kernel feature maps can be numerically tricky
- May struggle with exact copying or needle-in-a-haystack retrieval
- Less drop-in compatible with pretrained full-attention models

Comparison:

```text
Full attention:
  strong retrieval behavior
  expensive O(T^2)

Linear attention:
  cheaper for long T
  different inductive bias
  may lose exact pairwise token interactions
```

Interview answer:

> Linear attention trades exact full pairwise attention for better scaling. It can be much cheaper at long context, but quality and retrieval behavior may suffer depending on the method and task.

## 40. What Is Ring Attention?

Ring Attention is a distributed attention method for very long sequences.

The idea:

- Split the sequence across devices.
- Each device owns a chunk of queries, keys, and values.
- Devices pass key/value blocks around a ring.
- Each query chunk accumulates attention outputs over all key/value chunks.

Suppose sequence is split across `num_devices`.

```python
# On each device r:
q_local = q_chunk[r]  # [B, H, T_local, D]
k_local = k_chunk[r]  # [B, H, T_local, D]
v_local = v_chunk[r]  # [B, H, T_local, D]
```

Conceptual ring:

```python
# Pseudocode only.
k_block = k_local  # [B, H, T_local, D]
v_block = v_local  # [B, H, T_local, D]

running_out = 0  # [B, H, T_local, D]

for step in range(num_devices):
    scores = q_local @ k_block.transpose(-2, -1)
    # scores: [B, H, T_local, T_local]

    running_out = update_attention_output(running_out, scores, v_block)
    # running_out: [B, H, T_local, D]

    k_block, v_block = send_to_next_device_and_receive_previous(k_block, v_block)
```

Why it helps:

- no single GPU needs the full sequence
- long attention can be distributed across devices
- useful for training with very long contexts

Tradeoff:

- communication complexity
- implementation complexity
- synchronization overhead

Commentary: Ring Attention is mainly a systems technique for distributed long-context attention. It does not remove the conceptual cost of comparing many query/key pairs, but it makes the workload feasible across devices.

## 41. What Is Position Interpolation?

Position interpolation is a method for extending context length by mapping longer positions into the range seen during training.

Suppose a model was trained with context length:

```text
old_context = 8192
```

You want:

```text
new_context = 131072
```

Instead of feeding raw position `pos`, scale it down:

```python
def interpolate_positions(T_new, T_old, device):
    # T_new: new long context length
    # T_old: original trained context length
    scale = T_old / T_new
    pos = torch.arange(T_new, device=device).float()  # [T_new]
    return pos * scale  # [T_new], mapped into roughly [0, T_old]
```

For RoPE, this means using scaled positions when computing angles:

```python
def rope_cache_interpolated(T_new, dim, T_old, device, base=10000):
    # dim: D
    pos = interpolate_positions(T_new, T_old, device)  # [T_new]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # inv_freq: [D/2]

    angles = pos[:, None] * inv_freq[None, :]  # [T_new, D/2]
    return torch.cos(angles), torch.sin(angles)  # each [T_new, D/2]
```

Why it helps:

- avoids asking the model to handle completely unseen position frequencies
- compresses longer contexts into the trained positional range

Tradeoff:

- nearby positions become closer in positional space
- resolution may degrade
- often needs continued training or careful scaling

## 42. What Is YaRN?

YaRN stands for "Yet another RoPE extensioN" and is a RoPE scaling method for extending context length.

At a high level, YaRN modifies RoPE frequencies/scaling so a model can handle longer contexts better than naive extrapolation or simple interpolation.

The intuition:

- Some RoPE frequencies are more important for local position resolution.
- Some frequencies control long-range behavior.
- YaRN applies a more careful scaling strategy rather than uniformly compressing every frequency in the same way.

Very simplified conceptual pseudocode:

```python
def yarn_like_rope_cache(T_new, dim, device, base=10000, scale=16.0):
    # This is conceptual, not the exact YaRN formula.
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # inv_freq: [D/2]

    # Low frequencies may be scaled differently from high frequencies.
    frequency_id = torch.arange(inv_freq.numel(), device=device).float()  # [D/2]
    mix = (frequency_id / max(inv_freq.numel() - 1, 1)).clamp(0, 1)       # [D/2]

    scaled_inv_freq = inv_freq / (1.0 + mix * (scale - 1.0))  # [D/2]

    pos = torch.arange(T_new, device=device).float()  # [T_new]
    angles = pos[:, None] * scaled_inv_freq[None, :]  # [T_new, D/2]
    return torch.cos(angles), torch.sin(angles)       # each [T_new, D/2]
```

Commentary: The code above is not the exact YaRN implementation. It is only meant to explain the idea: RoPE frequency scaling can be non-uniform and designed to preserve both local and long-range behavior. If asked in an interview, be honest if you do not remember the exact formula and explain the purpose clearly.

Interview answer:

> YaRN is a RoPE scaling method for context extension. It tries to preserve short-context behavior while enabling longer-context extrapolation better than naive position interpolation.

## 43. How Would You Extend an 8k Model to 128k?

A practical plan:

### Step 1: Define the target behavior

Do you need:

- long-document retrieval?
- long-context reasoning?
- summarization?
- codebase understanding?
- multi-document QA?
- simply longer prompts without crashing?

Different goals require different data and evaluations.

### Step 2: Choose a positional strategy

Options:

- RoPE interpolation
- NTK-style RoPE scaling
- YaRN-style scaling
- continued pretraining with long sequences

Example:

```python
# Original context: 8k
T_old = 8192

# Target context: 128k
T_new = 131072

cos, sin = rope_cache_interpolated(
    T_new=T_new,
    dim=D,
    T_old=T_old,
    device=device,
)  # each [131072, D/2]
```

### Step 3: Use memory-efficient attention

For full attention at 128k, use kernels and parallelism:

- FlashAttention variants
- sequence parallelism
- context parallelism
- Ring Attention
- gradient checkpointing

### Step 4: Continue training

Use a mixture:

```text
short examples: preserve normal capability
medium examples: bridge the distribution
long examples: teach long-context behavior
```

Training batch sketch:

```python
# input_ids: [B, T], where T may vary by batch
# Some batches use T=8k, others 32k, 64k, 128k.
logits = model(input_ids)  # [B, T, Vocab]
```

### Step 5: Evaluate carefully

Do not only run needle-in-a-haystack.

Evaluate:

- retrieval at different depths
- multi-needle retrieval
- long-document QA
- long-context reasoning
- summarization
- robustness to distractors
- short-context regression

Interview answer:

> I would not just change a config. I would adjust positional encoding, use efficient attention systems, continue training on a length mixture, and evaluate retrieval, reasoning, summarization, and short-context regressions.

## 44. What Breaks When Context Becomes 1M Tokens?

At 1M tokens, many assumptions break.

### Attention compute

Full attention is enormous:

```python
# scores: [B, H, 1_000_000, 1_000_000]
```

Even with memory-efficient kernels, the compute is massive.

### KV cache

During inference:

```python
# k_cache, v_cache per layer: [B, H_kv, T_cache, D]
```

At `T_cache = 1_000_000`, KV cache memory can become huge.

### Position encoding

RoPE or other positional methods may degrade far beyond trained lengths.

### Data

There may be limited high-quality training data that actually requires 1M-token reasoning.

### Evaluation

Needle retrieval is not enough. A model can retrieve a fact but fail to reason across a million-token context.

### Optimization

Long batches are expensive and may reduce batch diversity. Training can become unstable or inefficient.

### Serving

Latency and cost become major issues:

```text
prefill cost: huge
decode KV reads: huge
memory pressure: huge
```

Interview answer:

> At 1M tokens, attention compute, KV cache memory, positional extrapolation, data quality, evaluation design, and serving latency all become bottlenecks. The problem becomes both algorithmic and systems-level.

## 45. How Would You Evaluate Long-Context Ability?

Long-context evaluation should test more than "can the model find a sentence?"

Use multiple categories.

### 1. Needle-in-a-haystack retrieval

Place a fact at different depths:

```text
context length: 8k, 32k, 128k
needle position: beginning, middle, end
question: asks for the needle
```

This tests basic retrieval.

### 2. Multi-needle retrieval

Place several facts:

```text
Fact A near the beginning
Fact B in the middle
Fact C near the end
Question requires A + B + C
```

This tests whether the model can retrieve multiple pieces.

### 3. Long-context reasoning

The answer should require combining information.

Example:

```text
Document says:
  Alice's project depends on Bob's dataset.
  Bob's dataset was updated after the March audit.
  The March audit invalidated datasets missing field X.

Question:
  Is Alice's project affected, and why?
```

### 4. Distractor robustness

Add irrelevant but similar facts.

```text
Needle: The access code is RIVER-42.
Distractor: The backup access code was RIVER-24.
```

### 5. Long summarization

Ask for summaries that preserve important details and avoid hallucination.

### 6. Short-context regression

Check that long-context training did not hurt normal tasks:

```python
# Evaluate both long and short sequences.
short_logits = model(short_input_ids)  # [B, T_short, Vocab]
long_logits = model(long_input_ids)    # [B, T_long, Vocab]
```

Metrics:

- exact match for retrieval
- citation accuracy
- answer faithfulness
- reasoning correctness
- robustness to distractors
- latency and memory
- short-context benchmark retention

Good evaluation grid:

```text
lengths: 8k, 32k, 64k, 128k
positions: start, middle, end
tasks: retrieval, multi-hop reasoning, summarization
distractors: none, mild, hard
```

Interview answer:

> I would evaluate long context with retrieval, multi-needle retrieval, reasoning across distant facts, summarization, distractor robustness, and short-context regression. A single needle benchmark is not enough.

## Final Interview Checklist

You should now be able to answer all Section C questions:

- Attention is `O(n^2)` because every token compares with every other token.
- Context extension stresses compute, memory, position encoding, training data, and evaluation.
- FlashAttention computes exact attention with tiling and online softmax.
- FlashAttention is faster because it reduces HBM traffic.
- FlashAttention solves the memory IO bottleneck, not the `O(n^2)` compute class.
- Sliding-window attention restricts attention to local neighborhoods.
- Sparse attention uses selected query-key patterns instead of full attention.
- Linear attention avoids explicit `T x T` attention using kernel-style reformulations.
- Linear attention scales better but may hurt retrieval and exact pairwise interactions.
- Ring Attention distributes long attention across devices.
- Position interpolation maps longer positions into the trained positional range.
- YaRN is a RoPE scaling method for better context extension.
- Extending 8k to 128k requires positional changes, efficient systems, continued training, and evaluation.
- At 1M tokens, compute, KV cache, position encoding, data, eval, and serving break.
- Long-context ability should be evaluated with retrieval, reasoning, summarization, distractors, and short-context regression.
