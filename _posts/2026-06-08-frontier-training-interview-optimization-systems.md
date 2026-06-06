---
title: "Frontier Training Interview (2): Optimization, Scaling Laws, and Distributed Training"
date: 2026-06-08 12:00:00 +0000
series_order: 3
categories: [interview-prep]
tags: [llm, optimization, adamw, scaling-laws, distributed-training, fsdp, zero]
pin: false
math: false
permalink: /posts/frontier-training-interview-optimization-systems/
---

## Goal

This tutorial prepares you for the training side of a frontier-model interview:

- optimizers
- learning-rate schedules
- mixed precision
- loss spikes and divergence
- scaling laws
- distributed training
- debugging slow or unstable runs

The key interview skill is connecting math, systems, and empirical debugging.

## Step 1: Understand Adam

Adam keeps two exponential moving averages:

- first moment: average gradient
- second moment: average squared gradient

Pseudocode:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad.square()

m_hat = m / (1 - beta1 ** step)
v_hat = v / (1 - beta2 ** step)

param = param - lr * m_hat / (torch.sqrt(v_hat) + eps)
```

Intuition:

- `m` smooths noisy gradients
- `v` rescales each parameter by recent gradient magnitude
- parameters with consistently large gradients get smaller effective steps

Interview answer:

> Adam adapts the learning rate per parameter using estimates of the first and second gradient moments. This makes optimization more robust than plain SGD for transformer training.

## Step 2: Why AdamW?

Classic Adam often implements weight decay by adding `weight_decay * param` to the gradient. That couples weight decay with Adam's adaptive gradient scaling.

AdamW decouples weight decay:

```python
# Adam update
param = param - lr * adam_update

# Decoupled weight decay
param = param - lr * weight_decay * param
```

Why this matters:

- weight decay behaves more like true parameter shrinkage
- tuning is cleaner
- it works reliably in large transformer training

Minimal AdamW-style pseudocode:

```python
@torch.no_grad()
def adamw_step(params, state, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
    beta1, beta2 = betas

    for p in params:
        if p.grad is None:
            continue

        s = state[p]
        s["step"] += 1

        g = p.grad
        s["m"].mul_(beta1).add_(g, alpha=1 - beta1)
        s["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2)

        m_hat = s["m"] / (1 - beta1 ** s["step"])
        v_hat = s["v"] / (1 - beta2 ** s["step"])

        p.mul_(1 - lr * weight_decay)
        p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

## Step 3: Gradient Clipping

Gradient clipping limits the global gradient norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Why it helps:

- prevents rare large updates
- reduces catastrophic loss spikes
- helps early training and mixed precision stability

It does not fix a broken run by itself. If clipping is constantly active, investigate learning rate, data, loss scaling, initialization, or numerical overflow.

## Step 4: Learning-Rate Warmup and Cosine Decay

Transformers often need warmup because early gradients are noisy and activations are not calibrated.

Warmup gradually increases LR:

```python
if step < warmup_steps:
    lr = max_lr * step / warmup_steps
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

Interview answer:

> Warmup prevents unstable updates at the beginning of training. Cosine decay then lowers the learning rate smoothly as the model approaches a better basin.

## Step 5: Mixed Precision

Large LLMs are usually trained in mixed precision for speed and memory.

FP16 has limited exponent range and can overflow or underflow. BF16 has fewer mantissa bits but the same exponent range as FP32, making it much more stable.

Typical training step:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # input_ids: [B, T]
    # logits: [B, T, vocab_size]
    logits = model(input_ids)
    # labels: [B, T]
    # flatten to [B*T, vocab_size] and [B*T]
    loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Why BF16 over FP16:

- larger dynamic range
- less need for loss scaling
- better stability for large transformer training

## Step 6: Gradient Accumulation

If the target batch size does not fit in memory, split it into microbatches:

```python
optimizer.zero_grad(set_to_none=True)

for micro_x, micro_y in microbatches:
    # micro_x: [micro_B, T]
    # micro_y: [micro_B, T] or [micro_B, T-1], depending on the loss helper
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model_loss(micro_x, micro_y)  # scalar
        loss = loss / grad_accum_steps

    loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This approximates a larger batch without storing all activations at once.

## Step 7: Gradient Checkpointing

Activation memory can dominate training memory. Gradient checkpointing saves memory by discarding some activations and recomputing them during backward.

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # x: [B, T, d_model]
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)  # [B, T, d_model]
    return x
```

Tradeoff:

- lower memory
- more compute
- often worthwhile for large models or long contexts

## Step 8: Loss Spikes and Divergence

Common causes of loss spikes:

- learning rate too high
- bad data batch
- numerical overflow
- optimizer state corruption
- incorrect loss masking
- distributed synchronization bug
- unstable long-context batch
- activation or gradient norm explosion

Debugging checklist:

1. Save the offending batch.
2. Re-run it deterministically.
3. Log gradient norms, activation norms, loss scale, and token statistics.
4. Check labels and masks.
5. Compare single-GPU and distributed behavior.
6. Reduce LR and disable mixed precision to isolate numerical issues.

Pseudocode for basic instrumentation:

```python
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().square()
    return total.sqrt()

if step % log_every == 0:
    print({
        "loss": loss.item(),
        "grad_norm": grad_norm(model).item(),
        "lr": scheduler.get_last_lr()[0],
        "tokens": input_ids.numel(),
    })
```

## Step 9: Batch Size and Gradient Noise Scale

Increasing batch size reduces gradient noise, but only up to a point.

Very large batches can hurt because:

- fewer parameter updates for the same number of tokens
- worse generalization
- less beneficial gradient noise
- optimizer hyperparameters may need retuning

Gradient noise scale estimates how noisy gradients are and helps choose a compute-efficient batch size.

Interview answer:

> Batch size scaling is not free. Larger batches improve hardware utilization and reduce gradient noise, but beyond the critical batch size they give diminishing returns and may hurt optimization.

## Step 10: Scaling Laws

Scaling laws describe predictable relationships between model size, data size, compute, and loss.

Kaplan-style result:

- larger models improve predictably
- performance follows power laws
- many models were trained with too few tokens relative to parameter count

Chinchilla-style result:

- for fixed compute, train smaller models on more data than earlier practice suggested
- compute-optimal training roughly balances model size and token count

Interview framing:

> If compute is fixed, I should not automatically maximize parameter count. I need to choose parameters and tokens together to minimize expected loss under the compute budget.

Simple compute estimate:

```python
def training_flops(params, tokens):
    # Common back-of-envelope estimate for dense decoder-only transformers.
    return 6 * params * tokens

def tokens_for_compute(compute_budget, params):
    return compute_budget / (6 * params)
```

Scaling-law deviations can come from:

- data quality changes
- architecture changes
- optimizer changes
- context length changes
- tokenizer changes
- benchmark contamination
- insufficient training stability

Do scaling laws continue forever?

No one knows. They are empirical laws over observed regimes. They can break when data, architecture, compute, or task distribution changes.

## Step 11: Data Parallelism

Each GPU has a full model replica and sees different data.

After backward, gradients are averaged:

```python
for p in model.parameters():
    # p.grad: same shape as p on this rank
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    p.grad /= world_size
```

Pros:

- simple
- effective when model fits on each GPU

Cons:

- optimizer states and parameters are replicated
- all-reduce can become a bottleneck

## Step 12: Tensor Parallelism

Tensor parallelism splits individual matrix multiplications across GPUs.

Example: split an MLP projection by columns:

```python
# Full: y = x @ W
# x: [B*T, in_dim]
# W: [in_dim, out_dim]
# TP rank r owns W_r: [in_dim, out_dim / tp_size]
y_r = x @ W_r                 # [B*T, out_dim / tp_size]
y = all_gather_or_reduce(y_r) # [B*T, out_dim]
```

It is useful when a layer is too large for one GPU or when matmul throughput benefits from sharding.

Cost:

- communication inside each transformer layer
- more complex implementation

## Step 13: Pipeline Parallelism

Pipeline parallelism splits layers across devices.

GPU 0 owns early layers, GPU 1 owns later layers:

```text
microbatch 1: GPU0 layer group A -> GPU1 layer group B
microbatch 2: GPU0 layer group A -> GPU1 layer group B
```

Pipeline bubbles are idle periods when some devices wait for work.

You reduce bubbles with more microbatches, but too many microbatches can hurt efficiency or change memory behavior.

## Step 14: ZeRO and FSDP

Training memory includes:

- parameters
- gradients
- optimizer states
- activations

ZeRO reduces memory by sharding these across data-parallel ranks.

FSDP similarly shards parameters, gradients, and optimizer states, gathering parameters only when needed for computation.

Conceptually:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = build_model()
model = FSDP(model)
```

Why it helps:

- each GPU stores only part of the full training state
- enables much larger models with data-parallel-style programming

Tradeoff:

- more communication
- sensitivity to wrapping policy and overlap settings

## Step 15: All-Reduce Bottlenecks

All-reduce becomes expensive when communication cannot overlap with computation or when network bandwidth is saturated.

Symptoms:

- low GPU utilization
- high step-time variance
- profiler shows communication kernels dominating
- scaling efficiency drops as GPU count increases

Debugging tools:

- PyTorch profiler
- NCCL logs
- GPU utilization metrics
- network counters
- per-rank step time logs

## Step 16: Training Run Is 20% Slower Overnight

How to debug:

1. Check whether code, data, container, driver, or cluster allocation changed.
2. Compare step-time breakdown: data loading, forward, backward, optimizer, communication.
3. Check GPU clocks, thermals, ECC errors, and failed links.
4. Look for straggler ranks.
5. Inspect data pipeline latency and storage throughput.
6. Compare sequence length distribution.
7. Check if activation checkpointing, precision, or kernel selection changed.
8. Reproduce on a small fixed batch.

Good interview answer:

> I would first separate input-pipeline slowdown from GPU-compute slowdown from communication slowdown. Then I would compare profiler traces and per-rank timings against a known-good run.

## Interview Checklist

You should be able to explain:

- Adam vs AdamW
- why warmup and cosine decay are used
- why BF16 is preferred for large LLMs
- how gradient accumulation and checkpointing change memory and compute
- causes of loss spikes and divergence
- Chinchilla compute-optimal training
- DP, TP, PP, ZeRO, and FSDP
- how to debug a distributed training slowdown

For training-team interviews, this section is often more important than memorizing architecture trivia.
