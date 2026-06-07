---
title: "Section G: Inference and Serving"
date: 2026-06-12 12:00:00 +0000
series_order: 6
categories: [interview-prep]
tags: [llm, inference, serving, kv-cache, speculative-decoding, quantization]
pin: false
math: false
permalink: /posts/inference-serving/
---

## Goal

This is the sixth section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover inference and serving:

1. KV caching
2. Why KV caching speeds up decoding
3. Prefill and decode phases
4. Which phase is compute-bound
5. Which phase is memory-bound
6. What causes inference latency
7. Speculative decoding
8. Quantization
9. Why INT4 is challenging
10. Reducing inference cost by 50%

Throughout the code:

- `B` = batch size
- `T_prompt` = prompt length
- `T_cache` = number of tokens already cached
- `T_new` = generated tokens
- `C` = model width
- `H_q` = number of query heads
- `H_kv` = number of key/value heads
- `D` = head dimension
- `Vocab` = vocabulary size

## 81. What Is KV Caching?

KV caching stores the key and value tensors produced by previous tokens during autoregressive generation.

In a decoder-only transformer, each attention layer computes:

```python
# x: [B, T, C]
q = q_proj(x)  # [B, T, H_q * D]
k = k_proj(x)  # [B, T, H_kv * D]
v = v_proj(x)  # [B, T, H_kv * D]
```

Then reshapes:

```python
q = q.view(B, T, H_q, D).transpose(1, 2)   # [B, H_q, T, D]
k = k.view(B, T, H_kv, D).transpose(1, 2)  # [B, H_kv, T, D]
v = v.view(B, T, H_kv, D).transpose(1, 2)  # [B, H_kv, T, D]
```

During generation, old `K` and `V` do not change. So we store them.

```python
class KVCache:
    def __init__(self, num_layers):
        self.keys = [None for _ in range(num_layers)]
        self.values = [None for _ in range(num_layers)]
```

For one layer:

```python
def append_to_cache(cache_k, cache_v, k_new, v_new):
    # cache_k, cache_v: [B, H_kv, T_cache, D] or None
    # k_new, v_new: [B, H_kv, 1, D] for one new token
    if cache_k is None:
        return k_new, v_new  # [B, H_kv, 1, D]

    cache_k = torch.cat([cache_k, k_new], dim=2)  # [B, H_kv, T_cache + 1, D]
    cache_v = torch.cat([cache_v, v_new], dim=2)  # [B, H_kv, T_cache + 1, D]
    return cache_k, cache_v
```

Production systems usually preallocate KV cache memory instead of repeatedly calling `torch.cat`.

## 82. Why Does KV Caching Speed Up Decoding?

Without KV caching, generation recomputes the full prefix every step.

Naive decoding:

```python
tokens = prompt  # [B, T_prompt]

for _ in range(T_new):
    logits = model(tokens)              # [B, current_T, Vocab]
    next_token = sample(logits[:, -1])  # [B, 1]
    tokens = torch.cat([tokens, next_token], dim=-1)  # [B, current_T + 1]
```

At each step, the model recomputes keys and values for all previous tokens. That is wasteful.

With KV cache:

```python
def decode_one_token(x_t, cache_k, cache_v):
    # x_t: [B, 1, C], only the newest token representation
    q_t = q_proj(x_t).view(B, 1, H_q, D).transpose(1, 2)    # [B, H_q, 1, D]
    k_t = k_proj(x_t).view(B, 1, H_kv, D).transpose(1, 2)   # [B, H_kv, 1, D]
    v_t = v_proj(x_t).view(B, 1, H_kv, D).transpose(1, 2)   # [B, H_kv, 1, D]

    cache_k, cache_v = append_to_cache(cache_k, cache_v, k_t, v_t)
    # cache_k, cache_v: [B, H_kv, T_cache + 1, D]

    # If H_q > H_kv, a GQA kernel handles grouping without materializing this repeat.
    k_for_attn = repeat_kv_if_needed(cache_k)  # [B, H_q, T_cache + 1, D]
    v_for_attn = repeat_kv_if_needed(cache_v)  # [B, H_q, T_cache + 1, D]

    scores = q_t @ k_for_attn.transpose(-2, -1)  # [B, H_q, 1, T_cache + 1]
    weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H_q, 1, T_cache + 1]
    out = weights @ v_for_attn  # [B, H_q, 1, D]
    return out, cache_k, cache_v
```

Why this is faster:

- Previous keys/values are reused.
- Only the new token runs through projections for the current step.
- The model avoids recomputing attention inputs for the whole prefix.

Tradeoff:

- KV caching uses memory proportional to context length.

```text
KV cache per layer roughly scales as:
B * H_kv * T_cache * D * 2

The factor 2 is for K and V.
```

## 83. What Are Prefill and Decode Phases?

LLM serving has two main phases.

### Prefill

Prefill processes the prompt.

```python
# input_ids: [B, T_prompt]
logits, cache = model.prefill(input_ids)
# logits: [B, T_prompt, Vocab]
# cache per layer: K/V tensors [B, H_kv, T_prompt, D]
```

During prefill, all prompt tokens are available at once. The model can use large matrix multiplications efficiently.

### Decode

Decode generates new tokens one at a time.

```python
next_token = sample(logits[:, -1])  # [B, 1]

for _ in range(T_new):
    logits, cache = model.decode_step(next_token, cache)
    # logits: [B, 1, Vocab]
    # cache grows from [B, H_kv, T_cache, D] to [B, H_kv, T_cache + 1, D]
    next_token = sample(logits[:, -1])  # [B, 1]
```

Key difference:

```text
Prefill:
  many tokens processed in parallel

Decode:
  one token generated per step
```

## 84. Which Phase Is Compute-Bound?

Prefill is usually more compute-bound.

Why:

- It processes many tokens at once.
- Matrix multiplications are large.
- GPUs can reach high utilization.

Example:

```python
# Prefill attention
# q, k, v: [B, H, T_prompt, D]
scores = q @ k.transpose(-2, -1)  # [B, H, T_prompt, T_prompt]
```

The matmuls are large enough to keep the GPU busy.

Prefill cost grows with prompt length:

```text
full attention prefill: roughly O(T_prompt^2)
MLP/projection compute: roughly O(T_prompt)
```

Interview answer:

> Prefill is usually compute-bound because the model processes many prompt tokens in parallel with large matrix multiplications.

Commentary: Very long prefill can also hit memory limits, especially from attention and activation storage. But serving prefill is commonly described as more compute-bound than decode.

## 85. Which Phase Is Memory-Bound?

Decode is usually more memory-bound.

At each decode step:

- only one new token is processed
- matmuls are smaller
- the model must read the KV cache for all previous tokens

Decode attention:

```python
# q_t: [B, H_q, 1, D]
# k_cache: [B, H_kv, T_cache, D]
# v_cache: [B, H_kv, T_cache, D]

k_for_attn = repeat_kv_if_needed(k_cache)  # [B, H_q, T_cache, D]
v_for_attn = repeat_kv_if_needed(v_cache)  # [B, H_q, T_cache, D]

scores = q_t @ k_for_attn.transpose(-2, -1)  # [B, H_q, 1, T_cache]
out = torch.softmax(scores, dim=-1) @ v_for_attn  # [B, H_q, 1, D]
```

The GPU repeatedly reads:

```text
K cache: [B, H_kv, T_cache, D]
V cache: [B, H_kv, T_cache, D]
```

As `T_cache` grows, memory bandwidth becomes a bottleneck.

Interview answer:

> Decode is usually memory-bandwidth-bound because each generated token must read the growing KV cache, while the per-token compute is relatively small.

## 86. What Causes Inference Latency?

Inference latency can come from many sources.

Important metrics:

- time to first token
- inter-token latency
- total generation time
- throughput
- tail latency

Latency causes:

1. Long prompts increase prefill time.
2. Long outputs increase decode steps.
3. Large models require more compute and memory reads.
4. KV cache reads become expensive at long context.
5. Poor batching wastes GPU capacity.
6. Over-batching can increase queueing delay.
7. Tokenization/detokenization can add CPU overhead.
8. Network overhead matters in serving systems.
9. Quantization/dequantization kernels may be inefficient.
10. Sampling and logits processing can add overhead.

Simple timing skeleton:

```python
start = time.time()
logits, cache = model.prefill(input_ids)  # input_ids: [B, T_prompt]
prefill_time = time.time() - start

decode_times = []
token = sample(logits[:, -1])  # [B, 1]

for _ in range(T_new):
    start = time.time()
    logits, cache = model.decode_step(token, cache)  # logits: [B, 1, Vocab]
    token = sample(logits[:, -1])                    # [B, 1]
    decode_times.append(time.time() - start)
```

Serving systems must balance:

```text
low latency for one user
high throughput across many users
low cost per token
```

## 87. Explain Speculative Decoding

Speculative decoding uses a small draft model to propose tokens and a larger target model to verify them.

Naive target decoding:

```text
target model generates 1 token per expensive forward pass
```

Speculative decoding:

```text
draft model proposes K tokens
target model verifies those K tokens in one forward pass
```

Conceptual pseudocode:

```python
prefix = input_ids  # [B, T]

# Draft model proposes K tokens quickly.
draft_tokens = draft_model.generate(prefix, num_tokens=K)  # [B, K]

# Target model verifies prefix + draft tokens.
candidate = torch.cat([prefix, draft_tokens], dim=-1)  # [B, T + K]
target_logits = target_model(candidate)  # [B, T + K, Vocab]

accepted = []
for i in range(K):
    # target_logits for the position predicting draft_tokens[:, i]
    verify_logits = target_logits[:, T + i - 1, :]  # [B, Vocab]
    proposed = draft_tokens[:, i]                   # [B]

    if accept(proposed, verify_logits):
        accepted.append(proposed)
    else:
        replacement = sample(verify_logits)  # [B]
        accepted.append(replacement)
        break
```

Why it helps:

- The target model evaluates several candidate positions in parallel.
- If many draft tokens are accepted, fewer target decode passes are needed.
- Quality can remain exact under the right acceptance algorithm.

When it works well:

- draft model is much cheaper
- draft model predictions often match target model
- decode is the bottleneck

Commentary: The exact accept/reject rule matters. Some speculative decoding variants preserve exact target-model sampling distribution; simplified versions may not.

## 88. Explain Quantization

Quantization represents weights or activations with fewer bits.

Examples:

```text
FP32 -> 32 bits
FP16/BF16 -> 16 bits
INT8 -> 8 bits
INT4 -> 4 bits
```

Why it helps:

- smaller model memory
- lower memory bandwidth
- faster inference if kernels support it
- larger models fit on the same hardware

Simple symmetric INT8 weight quantization:

```python
def quantize_int8_per_tensor(w):
    # w: floating-point weight tensor, e.g. [out_dim, in_dim]
    scale = w.abs().max() / 127  # scalar
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)  # [out_dim, in_dim]
    return q, scale

def dequantize_int8(q, scale):
    # q: int8 tensor, e.g. [out_dim, in_dim]
    return q.float() * scale  # [out_dim, in_dim]
```

Per-channel quantization:

```python
def quantize_int8_per_channel(w):
    # w: [out_dim, in_dim]
    scale = w.abs().amax(dim=1, keepdim=True) / 127  # [out_dim, 1]
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)  # [out_dim, in_dim]
    return q, scale
```

Types:

- weight-only quantization
- activation quantization
- KV-cache quantization
- quantization-aware training
- post-training quantization

Production quantization depends heavily on hardware kernels.

## 89. Why Is INT4 Challenging?

INT4 uses only 4 bits per value.

That means only 16 possible values before scaling.

Why it is hard:

- Very low precision
- Outlier weights can dominate scale
- Activations are harder to quantize than weights
- Errors can accumulate across many layers
- Reasoning/code tasks may be sensitive to small logit changes
- Specialized kernels are needed for speedups

Simple group-wise INT4 idea:

```python
def quantize_int4_groupwise(w, group_size=128):
    # w: [out_dim, in_dim]
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0

    w_grouped = w.view(out_dim, in_dim // group_size, group_size)
    # w_grouped: [out_dim, num_groups, group_size]

    scale = w_grouped.abs().amax(dim=-1, keepdim=True) / 7
    # scale: [out_dim, num_groups, 1]

    q = torch.round(w_grouped / scale).clamp(-7, 7)
    # q: [out_dim, num_groups, group_size], values fit in signed INT4 range conceptually

    return q, scale
```

Why group-wise scaling helps:

- One scale for the whole tensor may be too coarse.
- One scale per smaller group handles local variation better.

Commentary: PyTorch does not store true packed INT4 in a normal tensor in this simple pseudocode. Real INT4 inference packs two 4-bit values per byte and uses specialized kernels.

Interview answer:

> INT4 is challenging because the representation is extremely low precision. Outliers, activation ranges, kernel support, and quality degradation all become harder than INT8.

## 90. How Would You Reduce Inference Cost by 50%?

Start with measurement. Do not guess.

Break down cost:

```text
prefill compute
decode memory bandwidth
KV cache memory
batching efficiency
model size
sampling overhead
CPU/network overhead
hardware utilization
```

Instrumentation:

```python
metrics = {
    "prompt_tokens": input_ids.numel(),  # B*T_prompt
    "generated_tokens": T_new,
    "prefill_time": prefill_time,
    "mean_decode_time": sum(decode_times) / len(decode_times),
    "p95_decode_time": percentile(decode_times, 95),
}
```

Then choose interventions based on bottleneck.

### If decode is memory-bound

Use:

- GQA/MQA
- KV-cache quantization
- shorter max generation length
- better batching
- faster attention kernels

KV cache reduction:

```text
MHA: H_kv = H_q
GQA: H_kv < H_q
MQA: H_kv = 1
```

### If model weights dominate

Use:

- INT8 or INT4 weight quantization
- distillation to a smaller model
- pruning if appropriate
- better serving hardware

### If prefill dominates

Use:

- prompt caching
- prefix caching
- shorter prompts
- retrieval compression
- FlashAttention/paged attention kernels
- batching prefill requests

### If target decode is bottleneck

Use speculative decoding:

```python
draft_tokens = draft_model.generate(prefix, num_tokens=K)  # [B, K]
target_logits = target_model(torch.cat([prefix, draft_tokens], dim=-1))
# target_logits: [B, T + K, Vocab]
```

### If batching is poor

Use:

- continuous batching
- paged KV cache
- request scheduling
- separate prefill and decode scheduling

Good interview answer:

> I would first profile whether cost is dominated by prefill compute, decode memory bandwidth, model weights, KV cache, or batching inefficiency. Then I would combine targeted optimizations such as quantization, GQA/KV-cache reduction, speculative decoding, prompt caching, and better batching. The right path depends on the measured bottleneck.

Example 50% plan:

```text
1. Add INT8 or INT4 weight-only quantization: 20-40% cost reduction
2. Add continuous batching / scheduling improvements: 10-30%
3. Add speculative decoding for decode-heavy traffic: 20-50% on suitable workloads
4. Add prompt/prefix caching for repeated prompts: workload-dependent
```

Commentary: Percentages are workload-dependent. A 50% reduction is realistic in many systems, but only after profiling.

## Final Interview Checklist

You should now be able to answer all Section G questions:

- KV caching stores previous keys and values for reuse.
- KV caching speeds decoding by avoiding recomputation of previous tokens.
- Prefill processes the prompt; decode generates one token at a time.
- Prefill is usually compute-bound.
- Decode is usually memory-bandwidth-bound.
- Latency comes from prompts, generation length, KV cache, batching, kernels, CPU, and network overhead.
- Speculative decoding uses a cheap draft model and verifies with the target model.
- Quantization stores weights/activations in fewer bits.
- INT4 is hard because precision is extremely limited and kernels/quality become challenging.
- Cost reduction should start with profiling, then target the bottleneck with quantization, batching, KV-cache reduction, speculative decoding, and caching.
