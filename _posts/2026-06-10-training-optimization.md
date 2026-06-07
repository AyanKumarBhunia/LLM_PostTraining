---
title: "Section D: Training and Optimization"
date: 2026-06-10 12:00:00 +0000
series_order: 4
categories: [interview-prep]
tags: [llm, optimization, adamw, warmup, gradient-clipping, bf16, checkpointing]
pin: false
math: false
permalink: /posts/training-optimization/
---

## Goal

This is the fourth section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover training and optimization:

1. Adam
2. AdamW
3. Decoupled weight decay
4. Gradient clipping
5. Learning-rate warmup
6. Cosine learning-rate decay
7. Loss spikes
8. Training divergence
9. Gradient noise scale
10. Batch-size scaling
11. Why very large batches can hurt
12. Mixed precision training
13. BF16 vs FP16
14. Gradient checkpointing
15. Activation recomputation tradeoffs

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width
- `Vocab` = vocabulary size
- `P` = number of parameters in one tensor
- `micro_B` = microbatch size

## 46. Explain Adam

Adam is an adaptive optimizer. It keeps moving averages of:

- the gradient
- the squared gradient

For each parameter tensor:

```python
# p: [P] or any parameter shape, e.g. [out_dim, in_dim]
# grad: same shape as p

m = beta1 * m + (1 - beta1) * grad             # first moment, same shape as p
v = beta2 * v + (1 - beta2) * grad.square()    # second moment, same shape as p

m_hat = m / (1 - beta1 ** step)                # bias-corrected first moment
v_hat = v / (1 - beta2 ** step)                # bias-corrected second moment

p = p - lr * m_hat / (torch.sqrt(v_hat) + eps) # updated parameter, same shape as p
```

Intuition:

- `m` estimates the average direction of the gradient.
- `v` estimates the recent scale of gradients for each parameter.
- Parameters with large gradient variance get smaller effective updates.

Minimal Adam step:

```python
@torch.no_grad()
def adam_step(p, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    # p, grad: same shape, e.g. [P]
    state["step"] += 1

    state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)  # [P]
    state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # [P]

    m_hat = state["m"] / (1 - beta1 ** state["step"])  # [P]
    v_hat = state["v"] / (1 - beta2 ** state["step"])  # [P]

    p.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
    return p  # [P]
```

Interview answer:

> Adam adapts the update size per parameter using estimates of the first and second moments of gradients. This makes transformer optimization more stable than plain SGD in many settings.

## 47. Why AdamW?

AdamW is Adam with decoupled weight decay.

Classic Adam often applies weight decay by adding a penalty into the gradient:

```python
grad = grad + weight_decay * p  # same shape as p
```

Then Adam rescales that combined gradient by its adaptive denominator. This means weight decay is affected by Adam's per-parameter adaptation.

AdamW separates the two operations:

```python
# Adam update
p = p - lr * adam_update  # same shape as p

# Weight decay update
p = p - lr * weight_decay * p  # same shape as p
```

Why AdamW is used in LLMs:

- Weight decay behaves more like true parameter shrinkage.
- The learning rate and weight decay are easier to tune separately.
- It is empirically strong for transformer training.

Minimal AdamW:

```python
@torch.no_grad()
def adamw_step(params, state, lr, weight_decay=0.1, beta1=0.9, beta2=0.95, eps=1e-8):
    for p in params:
        if p.grad is None:
            continue

        # p: arbitrary shape, e.g. [out_dim, in_dim]
        # g: same shape as p
        g = p.grad
        s = state[p]
        s["step"] += 1

        s["m"].mul_(beta1).add_(g, alpha=1 - beta1)       # same shape as p
        s["v"].mul_(beta2).addcmul_(g, g, value=1 - beta2) # same shape as p

        m_hat = s["m"] / (1 - beta1 ** s["step"])  # same shape as p
        v_hat = s["v"] / (1 - beta2 ** s["step"])  # same shape as p

        p.mul_(1 - lr * weight_decay)  # decoupled weight decay
        p.addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
```

## 48. Why Decouple Weight Decay?

Weight decay is meant to directly discourage large weights.

With coupled weight decay:

```python
g = grad + weight_decay * p  # same shape as p
adam_update = g / adaptive_scale
```

The decay term gets divided by Adam's adaptive scale. So two parameters with the same value can decay differently depending on their gradient history.

With decoupled weight decay:

```python
p = p * (1 - lr * weight_decay)
```

Every parameter is shrunk directly and consistently.

Why this matters:

- The regularization effect is clearer.
- Hyperparameter tuning is cleaner.
- The optimizer's adaptive gradient scaling does not distort weight decay.

Interview answer:

> Decoupling weight decay keeps regularization separate from Adam's adaptive gradient update. This makes weight decay behave like direct parameter shrinkage rather than another gradient term that gets adaptively rescaled.

## 49. Explain Gradient Clipping

Gradient clipping limits the size of the gradient update.

The common version is global norm clipping:

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Conceptually:

```python
def global_grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            # p.grad: same shape as p
            total += p.grad.detach().float().norm(2).square()
    return total.sqrt()

def clip_grads(parameters, max_norm):
    norm = global_grad_norm(parameters)
    scale = max_norm / (norm + 1e-6)
    scale = min(scale, 1.0)

    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)  # same shape as p.grad
```

Why it helps:

- Prevents rare huge updates.
- Helps with loss spikes.
- Reduces risk of numerical instability.
- Useful during early training when gradients can be noisy.

Important:

> Gradient clipping is a safety mechanism, not a cure for a bad training setup.

If clipping triggers constantly, investigate:

- learning rate
- data quality
- loss masking
- numerical precision
- initialization
- optimizer settings

## 50. Why Do Transformers Need Learning-Rate Warmup?

At the beginning of training, transformer activations and optimizer states are not well calibrated.

If the learning rate starts too high, early updates can be unstable.

Warmup gradually increases the learning rate:

```python
def warmup_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr
```

Training loop:

```python
for step, batch in enumerate(loader):
    lr = warmup_lr(step, warmup_steps, max_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr

    # input_ids: [B, T]
    logits = model(input_ids)  # [B, T, Vocab]
    loss = loss_fn(logits, labels)  # scalar
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

Why warmup helps:

- Adam's moment estimates are initially biased/noisy.
- Deep residual networks can be sensitive to early update scale.
- Mixed precision can make early instability worse.
- Warmup reduces the chance of immediate divergence.

Interview answer:

> Transformers often need warmup because early optimization is fragile. Warmup lets optimizer states and activation scales stabilize before using the full learning rate.

## 51. Explain Cosine Learning-Rate Decay

Cosine decay smoothly reduces learning rate from a maximum to a minimum.

```python
def cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)
```

Shape note:

```python
# lr is scalar.
# It affects every parameter tensor through optimizer updates.
```

Why cosine decay is used:

- High learning rate early encourages fast progress.
- Lower learning rate later helps refine the solution.
- The transition is smooth.
- It is simple and works well empirically.

Typical schedule:

```text
warmup -> peak learning rate -> smooth decay -> final low learning rate
```

Commentary: Cosine decay is not the only valid schedule. Some training runs use linear decay, constant LR, inverse square root, or schedule variants. Cosine is a strong common default.

## 52. What Causes Loss Spikes?

A loss spike is a sudden jump in training loss.

Common causes:

- Bad or corrupted data batch
- Learning rate too high
- Gradient explosion
- Numerical overflow
- Incorrect loss masking
- Very long or unusual sequence batch
- Optimizer state issue
- Distributed training bug
- Mixed precision instability

Basic instrumentation:

```python
def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # p.grad: same shape as p
            total += p.grad.detach().float().norm().square()
    return total.sqrt()

if step % log_every == 0:
    # input_ids: [B, T]
    print({
        "step": step,
        "loss": loss.item(),
        "grad_norm": grad_norm(model).item(),
        "lr": optimizer.param_groups[0]["lr"],
        "tokens": input_ids.numel(),  # B*T
        "seq_len": input_ids.shape[1],
    })
```

Debugging plan:

1. Save the batch that caused the spike.
2. Re-run it deterministically.
3. Check tokenization and labels.
4. Check loss masks.
5. Log gradient and activation norms.
6. Try lower learning rate.
7. Try FP32/BF16 changes to isolate numerical issues.

Interview answer:

> Loss spikes usually come from unstable update scale, bad data, masking bugs, or numerical issues. I would reproduce the offending batch and inspect data, masks, gradients, activations, and precision.

## 53. What Causes Training Divergence?

Training divergence means the run does not recover. Loss may become `NaN`, explode, or stay unstable.

Common causes:

- Learning rate far too high
- Warmup too short
- Broken optimizer configuration
- Incorrect loss computation
- Incorrect causal or padding mask
- Bad initialization
- FP16 overflow
- Unclipped gradient explosion
- Data corruption
- Distributed all-reduce or sharding bug

Detection:

```python
if not torch.isfinite(loss):
    # loss: scalar
    print("non-finite loss at step", step)
    save_debug_batch(batch)
    raise RuntimeError("training diverged")

for name, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        # p.grad: same shape as p
        print("non-finite grad:", name)
        raise RuntimeError("bad gradient")
```

Difference between spike and divergence:

- A spike may recover.
- Divergence keeps getting worse or becomes non-finite.

Interview answer:

> Divergence is usually a persistent instability from bad hyperparameters, numerical overflow, incorrect objective/masks, or distributed bugs. I would first check finiteness, LR schedule, gradient norms, and whether the loss computation is correct.

## 54. What Is Gradient Noise Scale?

Gradient noise scale describes how noisy stochastic gradients are relative to the true full-batch gradient.

Small batch:

```text
more noisy gradient
more frequent updates
```

Large batch:

```text
less noisy gradient
fewer updates for same token budget
```

Conceptual estimate:

```python
# grads: [num_microbatches, P]
# Each row is a flattened gradient estimate from one microbatch.
mean_grad = grads.mean(dim=0)  # [P]
noise = grads - mean_grad[None, :]  # [num_microbatches, P]

signal = mean_grad.norm().square()  # scalar
variance = noise.pow(2).sum(dim=1).mean()  # scalar

gradient_noise_scale = variance / (signal + 1e-12)
```

Why it matters:

- It helps estimate the critical batch size.
- Below the critical batch size, increasing batch size improves efficiency.
- Above it, larger batches give diminishing returns.

Commentary: Exact gradient noise scale estimation can be more subtle than this toy code. In interviews, focus on the concept: gradient noise determines how useful increasing batch size will be.

## 55. What Is Batch-Size Scaling?

Batch-size scaling studies how training changes as batch size increases.

There are two common meanings:

### 1. Hardware scaling

Increase global batch size to use more GPUs efficiently.

```python
global_batch = micro_batch * grad_accum_steps * data_parallel_world_size
```

Example:

```python
micro_batch = 4
grad_accum_steps = 8
world_size = 64

global_batch = micro_batch * grad_accum_steps * world_size  # 2048 sequences
```

### 2. Optimization scaling

Adjust learning rate and schedule as batch size changes.

Common heuristic:

```text
larger batch -> larger learning rate, up to a point
```

Gradient accumulation:

```python
optimizer.zero_grad(set_to_none=True)

for micro in range(grad_accum_steps):
    # input_ids: [micro_B, T]
    logits = model(input_ids)  # [micro_B, T, Vocab]
    loss = loss_fn(logits, labels)  # scalar
    loss = loss / grad_accum_steps
    loss.backward()

optimizer.step()
```

Dividing by `grad_accum_steps` keeps the effective gradient scale comparable to one large batch.

## 56. Why Can Very Large Batches Hurt?

Very large batches can hurt because they reduce gradient noise too much and reduce the number of optimizer updates for a fixed token budget.

Problems:

- Diminishing returns past the critical batch size
- Worse generalization in some regimes
- Fewer parameter updates per token budget
- More sensitivity to learning-rate tuning
- Less beneficial stochasticity
- Larger batches may require more memory and communication

Example:

```text
Train on 1T tokens.

Batch A:
  1M tokens per step -> 1,000,000 updates

Batch B:
  10M tokens per step -> 100,000 updates
```

Batch B has fewer chances to update parameters.

Interview answer:

> Very large batches can hurt because after the critical batch size, extra examples mostly reduce noise without giving proportional optimization benefit. You also get fewer updates for the same token budget and may need careful LR retuning.

## 57. Explain Mixed Precision Training

Mixed precision training uses lower-precision dtypes for speed and memory while keeping some operations or states in higher precision.

Common setup:

- Forward activations in BF16 or FP16
- Some reductions in FP32
- Optimizer states often in FP32
- Master weights may be FP32 depending on setup

PyTorch-style training step:

```python
for batch in loader:
    input_ids = batch["input_ids"]  # [B, T]
    labels = batch["labels"]        # [B, T]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)  # [B, T, Vocab], computed with mixed precision kernels
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
            labels.reshape(-1),                   # [B*T]
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

Why it helps:

- Lower activation memory
- Faster matmuls on modern accelerators
- Larger batch or sequence length may fit
- Lower bandwidth pressure

Commentary: With FP16, loss scaling is often needed. With BF16, loss scaling is usually less important because BF16 has a much larger exponent range.

## 58. Why Use BF16 Instead of FP16?

FP16 and BF16 are both 16-bit formats, but they allocate bits differently.

FP16:

- more mantissa precision than BF16
- smaller exponent range
- more likely to overflow/underflow

BF16:

- same exponent range as FP32
- less mantissa precision
- much more stable for large-scale training

Why BF16 is preferred for LLMs:

- Handles large/small activation and gradient values better
- Reduces need for dynamic loss scaling
- More robust in deep transformers
- Supported efficiently on modern accelerators

Example:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # x: [B, T, C]
    logits = model(input_ids)  # [B, T, Vocab]
    loss = loss_fn(logits, labels)  # scalar
```

Interview answer:

> BF16 is usually preferred because it has FP32-like exponent range, making it much less prone to overflow than FP16. For large transformer training, that stability is often more valuable than FP16's extra mantissa bits.

## 59. Explain Gradient Checkpointing

Gradient checkpointing saves memory by not storing all activations during the forward pass.

Normally, backprop needs activations:

```python
# x: [B, T, C]
y = block(x)  # [B, T, C]
# Autograd stores intermediate activations inside block for backward.
```

With checkpointing:

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # x: [B, T, C]
    for block in self.blocks:
        x = checkpoint(block, x, use_reentrant=False)  # [B, T, C]
    return x
```

Instead of storing all intermediate activations, PyTorch stores only selected inputs and recomputes the missing forward activations during backward.

Why it helps:

- Lower activation memory
- Enables larger models, batches, or contexts
- Useful for long-context training where activations are huge

Cost:

- More compute
- Slower training step
- More complicated debugging/profiling

## 60. What Are Activation Re-computation Tradeoffs?

Activation recomputation is the compute-memory tradeoff behind gradient checkpointing.

Without checkpointing:

```text
Forward:
  compute activations
  store many activations

Backward:
  reuse stored activations
```

With checkpointing:

```text
Forward:
  compute activations
  store fewer activations

Backward:
  recompute some activations
  then compute gradients
```

Tradeoffs:

- Saves memory
- Costs extra compute
- Can reduce throughput
- May allow larger batch size or sequence length
- Can improve overall hardware utilization if memory was the bottleneck

Toy timing intuition:

```python
# x: [B, T, C]

# No checkpoint:
y = block(x)  # stores internal activations, faster backward, more memory

# Checkpoint:
y = checkpoint(block, x, use_reentrant=False)  # stores less, recomputes in backward
```

When it is worth it:

- activation memory dominates
- long context training
- model barely does not fit
- you can trade extra compute for feasible training

When it may not be worth it:

- training is already compute-bound
- memory is not the bottleneck
- recomputation slows the run too much

Interview answer:

> Activation recomputation trades extra forward compute during backward for lower activation memory. It is often worthwhile for large models or long contexts, but it can reduce throughput.

## Final Interview Checklist

You should now be able to answer all Section D questions:

- Adam tracks first and second gradient moments.
- AdamW decouples adaptive optimization from weight decay.
- Decoupled weight decay gives direct parameter shrinkage.
- Gradient clipping limits rare large updates.
- Warmup stabilizes early transformer training.
- Cosine decay smoothly lowers LR after warmup.
- Loss spikes can come from data, LR, masks, gradients, or numerics.
- Divergence is persistent instability or non-finite training.
- Gradient noise scale describes noise relative to gradient signal.
- Batch-size scaling changes optimization and hardware efficiency.
- Very large batches can give diminishing returns and fewer updates.
- Mixed precision saves memory and speeds up matmuls.
- BF16 is preferred because its exponent range is much safer than FP16.
- Gradient checkpointing stores fewer activations and recomputes them.
- Activation recomputation trades compute for memory.
