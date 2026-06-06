---
title: "Frontier Training Interview (3): Inference, Serving, and Research Experiments"
date: 2026-06-09 12:00:00 +0000
series_order: 4
categories: [interview-prep]
tags: [llm, inference, kv-cache, speculative-decoding, quantization, evaluation, research]
pin: false
math: false
permalink: /posts/frontier-training-interview-inference-research/
---

## Goal

This tutorial prepares you for two interview areas that often separate strong candidates from merely theoretical ones:

1. Can you reason about inference bottlenecks?
2. Can you design experiments that produce trustworthy conclusions?

Frontier training teams care about both. A model is only useful if it can be trained, evaluated, served, and improved reliably.

## Step 1: Understand Autoregressive Decoding

A decoder-only LLM generates one token at a time:

```python
tokens = prompt  # [B, prompt_T]

for _ in range(max_new_tokens):
    logits = model(tokens)              # [B, current_T, vocab_size]
    next_token = sample(logits[:, -1])  # [B, 1], sampled from last-position logits
    tokens = torch.cat([tokens, next_token], dim=-1)  # [B, current_T + 1]
```

This naive version recomputes all previous hidden states at every step. That is extremely wasteful.

## Step 2: KV Caching

Each transformer layer computes keys and values for previous tokens. During decoding, previous keys and values do not change.

KV caching stores them:

```python
def attention_step(x_t, cache):
    # x_t: [B, 1, d_model] for one decode step
    q_t, k_t, v_t = project_qkv(x_t)
    # q_t: [B, H_q, 1, D]
    # k_t, v_t: [B, H_kv, 1, D]

    cache.k = torch.cat([cache.k, k_t], dim=2)  # [B, H_kv, cached_T, D]
    cache.v = torch.cat([cache.v, v_t], dim=2)  # [B, H_kv, cached_T, D]

    scores = q_t @ cache.k.transpose(-2, -1)    # [B, H_q, 1, cached_T]
    weights = torch.softmax(scores / math.sqrt(d_head), dim=-1)
    y_t = weights @ cache.v                      # [B, H_q, 1, D]
    return y_t, cache
```

Why KV caching speeds up decoding:

- previous tokens do not need to be recomputed
- each new step only computes projections for the new token
- attention still reads previous K/V, but avoids full forward recomputation

The tradeoff is memory. KV cache size grows with:

```text
batch_size * layers * kv_heads * sequence_length * head_dim
```

This is why GQA and MQA matter for inference.

## Step 3: Prefill vs Decode

LLM serving has two phases:

### Prefill

The model processes the input prompt.

Characteristics:

- many tokens processed in parallel
- large matrix multiplications
- usually compute-bound
- benefits from high GPU utilization

### Decode

The model generates one token at a time.

Characteristics:

- sequential
- repeatedly reads KV cache
- often memory-bandwidth-bound
- latency-sensitive

Interview answer:

> Prefill is usually compute-bound because the model processes the prompt in parallel. Decode is often memory-bound because each new token reads the growing KV cache while doing relatively small matmuls.

## Step 4: What Causes Inference Latency?

Latency can come from:

- long prompts
- long generated outputs
- large batch queues
- slow tokenizer or detokenizer
- KV cache memory bandwidth
- inefficient attention kernels
- CPU scheduling overhead
- network overhead
- poor batching strategy
- model too large for target hardware

Serving systems optimize multiple metrics at once:

- time to first token
- inter-token latency
- total throughput
- cost per generated token
- tail latency

## Step 5: Sampling Methods

### Top-k Sampling

Keep only the top `k` logits:

```python
def top_k_sample(logits, k):
    # logits: [B, vocab_size]
    values, indices = torch.topk(logits, k)        # each [B, k]
    probs = torch.softmax(values, dim=-1)          # [B, k]
    sampled = torch.multinomial(probs, num_samples=1) # [B, 1]
    return indices.gather(-1, sampled)             # [B, 1]
```

### Nucleus Sampling

Keep the smallest set of tokens whose cumulative probability exceeds `p`:

```python
def nucleus_sample(logits, p=0.9):
    # logits: [B, vocab_size]
    probs = torch.softmax(logits, dim=-1)                 # [B, vocab_size]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    # sorted_probs, sorted_idx: [B, vocab_size]
    cumulative = torch.cumsum(sorted_probs, dim=-1)       # [B, vocab_size]

    keep = cumulative <= p                                # [B, vocab_size]
    keep[..., 0] = True

    filtered = sorted_probs.masked_fill(~keep, 0.0)       # [B, vocab_size]
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(filtered, num_samples=1)  # [B, 1]
    return sorted_idx.gather(-1, sampled)                 # [B, 1]
```

Interview framing:

> Top-k limits sampling to a fixed number of candidates. Nucleus sampling adapts the candidate set size based on probability mass.

## Step 6: Speculative Decoding

Speculative decoding uses a small draft model to propose multiple tokens. The large target model verifies them in parallel.

Conceptual algorithm:

```python
draft_tokens = draft_model.generate(prefix, num_tokens=k)  # [B, k]
target_logits = target_model(prefix + draft_tokens)        # [B, prefix_T + k, vocab]

accepted = []
for i, token in enumerate(draft_tokens):
    if verification_accepts(token, target_logits[i]):
        accepted.append(token)
    else:
        accepted.append(sample(target_logits[i]))
        break
```

Why it helps:

- the expensive model verifies several positions in one forward pass
- accepted draft tokens reduce the number of target-model decode steps

When it helps most:

- draft model is much faster
- draft acceptance rate is high
- target decode is the bottleneck

## Step 7: Quantization

Quantization stores weights or activations with fewer bits.

Examples:

- FP16 or BF16
- INT8
- INT4
- FP8

Why it reduces cost:

- lower memory footprint
- higher effective memory bandwidth
- sometimes faster matmuls on supported hardware

Why INT4 is challenging:

- less representational precision
- outlier channels can dominate error
- activation quantization is harder than weight-only quantization
- quality may degrade on reasoning or code tasks

Simple symmetric weight quantization:

```python
def quantize_int8(w):
    # w: any floating-point weight tensor, e.g. [out_dim, in_dim]
    scale = w.abs().max() / 127
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)  # same shape as w
    return q, scale

def dequantize_int8(q, scale):
    return q.float() * scale  # same shape as q
```

Production quantization is more sophisticated: per-channel scales, group-wise quantization, calibration, and specialized kernels.

## Step 8: Reducing Inference Cost by 50%

A practical answer should mention measurement first:

1. Break cost into prefill, decode, batching, memory, and infrastructure overhead.
2. Measure utilization and latency percentiles.
3. Identify whether the workload is compute-bound or memory-bound.

Possible interventions:

- use GQA or smaller KV cache
- quantize weights
- improve batching and scheduling
- use speculative decoding
- reduce max generation length
- distill to a smaller model
- cache repeated prompts
- optimize tokenizer and CPU overhead
- choose hardware better matched to workload

Good interview answer:

> I would not start with a single trick. I would profile the serving path, identify whether cost is dominated by prefill, decode, memory bandwidth, or poor batching, and then choose optimizations that target that bottleneck.

## Step 9: Evaluating a New Architecture

A good architecture experiment needs:

- a baseline
- a controlled training budget
- matched data
- matched tokenizer
- matched optimizer settings
- multiple seeds when affordable
- downstream evaluations
- ablations

Minimal experiment plan:

```text
Baseline: existing transformer block
Variant: transformer block with new attention/MLP/norm
Budget: same tokens, same compute, same data mixture
Metrics: validation loss, task evals, speed, memory, stability
Decision: improvement must beat noise and justify complexity
```

## Step 10: Knowing Whether an Improvement Is Real

An improvement is more credible when:

- it repeats across seeds
- it holds at multiple scales
- it improves validation loss and downstream metrics
- it does not only improve one benchmark
- it does not rely on extra compute or data
- it survives ablations

Bad signs:

- only one lucky run
- benchmark-only gain with worse validation loss
- unclear data contamination
- hyperparameter advantage over baseline
- higher cost hidden behind a better score

## Step 11: Ablations

An ablation removes or isolates one part of a proposed change.

Example for a new long-context method:

1. Baseline RoPE.
2. RoPE with interpolation only.
3. RoPE with YaRN-style scaling only.
4. Full method.
5. Full method with same compute but no long-context data.
6. Full method with long-context data but original position encoding.

The goal is to identify which part caused the gain.

## Step 12: Detecting Benchmark Overfitting

Signals:

- validation loss does not improve but benchmark does
- improvement appears only on a small benchmark family
- model gives memorized benchmark-style answers
- performance collapses under rephrasing
- private or newly generated tests do not improve

Mitigations:

- hold out private evaluations
- use contamination checks
- evaluate generated variants
- use human evaluation
- test capabilities, not only benchmark names

## Step 13: Debugging a Regression

If perplexity improved but coding benchmarks dropped, possible causes include:

- data mixture shifted away from code
- tokenizer change hurt code formatting
- post-training made outputs more verbose
- safety tuning suppressed code generation
- benchmark prompt format changed
- model became better at average text but worse at structured reasoning

Debugging plan:

1. Compare data mixtures.
2. Evaluate by domain.
3. Inspect qualitative samples.
4. Check prompt formatting.
5. Compare base, SFT, and preference-tuned checkpoints.
6. Run targeted code perplexity and pass-at-k evaluations.

## Step 14: Evaluating Reasoning Ability

Reasoning evaluation should include:

- math
- code
- multi-hop QA
- tool-use tasks
- adversarial variants
- out-of-distribution prompts
- process inspection when possible

Do not rely only on final accuracy. Track:

- calibration
- consistency under paraphrase
- sensitivity to irrelevant context
- ability to recover from mistakes
- performance as problem length increases

## Step 15: Long-Context Reasoning Experiment

A strong long-context experiment separates retrieval from reasoning.

Design:

1. Put several facts across a long document.
2. Ask questions that require combining facts.
3. Vary distance between facts.
4. Add distractors.
5. Compare short-context oracle, long-context model, and retrieval-augmented baseline.

Example:

```text
Document length: 8k, 32k, 128k
Task: answer requires fact A near start and fact B near end
Controls: same facts in short context
Metrics: exact match, explanation quality, citation correctness
```

This tells you whether the model can reason over long context, not just retrieve a needle.

## Step 16: Talking About Failed Research

A good answer has structure:

1. What hypothesis did you test?
2. Why was it plausible?
3. What experiment did you run?
4. What failed?
5. What did you learn?
6. What would you do differently?

Example framing:

> We thought a routing change would improve specialization. It improved training loss slightly but hurt downstream robustness. Ablations showed the gain came from extra capacity, not better routing. The lesson was to compare against capacity-matched baselines before attributing gains to the mechanism.

## Interview Checklist

You should be able to explain:

- KV caching
- prefill vs decode
- memory-bound vs compute-bound inference
- speculative decoding
- quantization tradeoffs
- how to reduce inference cost
- how to evaluate a new architecture
- what ablations to run
- how to detect benchmark overfitting
- how to debug regressions

The best answers sound like an experimental scientist and a systems engineer in the same person.
