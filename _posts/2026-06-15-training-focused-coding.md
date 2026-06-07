---
title: "Coding Exercises: Training-Focused Implementations"
date: 2026-06-15 12:00:00 +0000
series_order: 9
categories: [interview-prep]
tags: [llm, pytorch, distributed-training, adamw, mixed-precision, checkpointing]
pin: false
math: false
permalink: /posts/training-focused-coding/
---

## Goal

This tutorial covers training-focused coding exercises that are likely in a frontier-model Training Team interview.

We will implement:

1. Distributed training loop
2. Gradient accumulation
3. Mixed precision training
4. Cosine scheduler
5. AdamW
6. Gradient clipping
7. Checkpoint saving/loading
8. EMA weights
9. Activation checkpointing
10. Debugging a diverging training loop

Throughout the code:

- `B` = global or local batch size depending on context
- `micro_B` = microbatch size
- `T` = sequence length
- `C` = model width
- `Vocab` = vocabulary size
- `P` = parameter tensor size after flattening
- `world_size` = number of distributed processes

## 1. Write a Distributed Training Loop

The most common PyTorch starting point is Distributed Data Parallel, or DDP.

Each process:

- owns one GPU
- has a full model replica
- receives a shard of the data
- computes gradients locally
- synchronizes gradients with all-reduce

Minimal setup:

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), dist.get_rank()
```

Training loop:

```python
def train_ddp(model, dataset, optimizer, scheduler, epochs):
    local_rank, world_size, rank = setup_distributed()
    device = torch.device("cuda", local_rank)

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)

        for batch in loader:
            input_ids = batch["input_ids"].to(device)  # [local_B, T]
            labels = batch["labels"].to(device)        # [local_B, T]

            optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids)  # [local_B, T, Vocab]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [local_B*T, Vocab]
                labels.reshape(-1),                   # [local_B*T]
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    dist.destroy_process_group()
```

What DDP handles:

```text
After backward:
  each rank has local gradients
  DDP all-reduces gradients
  optimizer.step() applies the same averaged update on each rank
```

Commentary: Real frontier training often uses FSDP, ZeRO, tensor parallelism, pipeline parallelism, or custom stacks. DDP is still the cleanest interview baseline.

## 2. Implement Gradient Accumulation

Gradient accumulation simulates a larger batch by splitting it into microbatches.

Without accumulation:

```text
one forward/backward -> one optimizer step
```

With accumulation:

```text
several forward/backward passes -> one optimizer step
```

Implementation:

```python
def train_with_accumulation(model, loader, optimizer, scheduler, grad_accum_steps):
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].cuda()  # [micro_B, T]
        labels = batch["labels"].cuda()        # [micro_B, T]

        logits = model(input_ids)  # [micro_B, T, Vocab]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [micro_B*T, Vocab]
            labels.reshape(-1),                   # [micro_B*T]
        )

        loss = loss / grad_accum_steps  # scalar, keeps gradient scale correct
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
```

Effective batch size:

```python
effective_batch = micro_batch_size * grad_accum_steps * world_size
```

Why divide loss:

> Without dividing by `grad_accum_steps`, the accumulated gradient would be too large.

## 3. Implement Mixed Precision Training

Mixed precision uses lower precision for speed/memory while keeping training stable.

BF16 version:

```python
def train_mixed_precision_bf16(model, loader, optimizer, scheduler):
    model.train()

    for batch in loader:
        input_ids = batch["input_ids"].cuda()  # [B, T]
        labels = batch["labels"].cuda()        # [B, T]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)  # [B, T, Vocab]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
                labels.reshape(-1),                   # [B*T]
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
```

FP16 version often needs gradient scaling:

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    input_ids = batch["input_ids"].cuda()  # [B, T]
    labels = batch["labels"].cuda()        # [B, T]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(input_ids)  # [B, T, Vocab]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
            labels.reshape(-1),                   # [B*T]
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

Commentary: BF16 is usually preferred for LLM training on modern accelerators because it has a much wider exponent range than FP16.

## 4. Implement Cosine Scheduler

Cosine learning-rate decay usually follows warmup.

```python
def cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    # step: scalar integer
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)
```

Using it:

```python
for step, batch in enumerate(loader):
    lr = cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr)

    for group in optimizer.param_groups:
        group["lr"] = lr

    # training step...
```

As a PyTorch scheduler:

```python
class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        lr = cosine_lr(
            self.step_num,
            self.warmup_steps,
            self.total_steps,
            self.max_lr,
            self.min_lr,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.step_num += 1
```

## 5. Implement AdamW

AdamW is Adam with decoupled weight decay.

```python
class AdamWCustom:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

        for p in self.params:
            self.state[p] = {
                "step": 0,
                "m": torch.zeros_like(p),  # same shape as p
                "v": torch.zeros_like(p),  # same shape as p
            }

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad  # same shape as p
            s = self.state[p]
            s["step"] += 1

            s["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)       # same shape as p
            s["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2) # same shape as p

            m_hat = s["m"] / (1 - self.beta1 ** s["step"])  # same shape as p
            v_hat = s["v"] / (1 - self.beta2 ** s["step"])  # same shape as p

            # Decoupled weight decay.
            p.mul_(1 - self.lr * self.weight_decay)

            # Adam update.
            p.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
```

Commentary: This is for interview understanding. Production optimizers handle parameter groups, dtype states, fused kernels, distributed sharding, and many edge cases.

## 6. Implement Gradient Clipping

Global norm clipping:

```python
def global_grad_norm(parameters):
    total = torch.tensor(0.0, device="cuda")

    for p in parameters:
        if p.grad is not None:
            # p.grad: same shape as p
            grad = p.grad.detach().float()
            total += grad.norm(2).pow(2)

    return total.sqrt()  # scalar
```

Clip implementation:

```python
@torch.no_grad()
def clip_grad_norm(parameters, max_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    total_norm = global_grad_norm(params)

    clip_coef = max_norm / (total_norm + eps)
    clip_coef = torch.clamp(clip_coef, max=1.0)

    for p in params:
        p.grad.mul_(clip_coef)  # same shape as p.grad

    return total_norm
```

In a training loop:

```python
loss.backward()
grad_norm = clip_grad_norm(model.parameters(), max_norm=1.0)
optimizer.step()
```

What to explain:

- Clip after backward.
- Clip before optimizer step.
- If clipping is always active, debug LR/data/numerics.

## 7. Implement Checkpoint Saving and Loading

A useful training checkpoint includes:

- model weights
- optimizer state
- scheduler state
- step number
- RNG states if exact reproducibility matters
- scaler state for FP16 mixed precision

Saving:

```python
def save_checkpoint(path, model, optimizer, scheduler, step, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "step": step,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
    }

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)
```

Loading:

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cuda"):
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    torch.set_rng_state(checkpoint["rng_cpu"])
    torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])

    return checkpoint["step"]
```

Commentary: In distributed training, only rank 0 often writes checkpoint metadata, but model/optimizer states may be sharded with FSDP/ZeRO. Loading then requires the same distributed strategy or a conversion step.

## 8. Implement EMA Weights

EMA means exponential moving average of model weights.

It keeps a smoothed copy of parameters:

```python
ema = decay * ema + (1 - decay) * param
```

Implementation:

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()  # same shape as p

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            # p: parameter tensor, any shape
            self.shadow[name].mul_(self.decay)
            self.shadow[name].add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.copy_(self.shadow[name])  # same shape as p
```

Training use:

```python
ema = EMA(model, decay=0.999)

for batch in loader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    ema.update(model)
```

Why use EMA:

- smoother weights
- sometimes better validation/eval stability
- common in diffusion/vision; less universal in large LLM pretraining

Commentary: EMA is not always used for frontier LLM pretraining because maintaining another full copy of weights is expensive. But it is a common coding/interview exercise.

## 9. Implement Activation Checkpointing

Activation checkpointing saves memory by recomputing activations during backward.

Without checkpointing:

```text
forward stores many intermediate activations
backward reuses them
```

With checkpointing:

```text
forward stores fewer activations
backward recomputes missing activations
```

Implementation:

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(nn.Module):
    def __init__(self, blocks, norm, lm_head):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, x):
        # x: [B, T, C]
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)  # [B, T, C]

        x = self.norm(x)        # [B, T, C]
        logits = self.lm_head(x) # [B, T, Vocab]
        return logits
```

Selective checkpointing:

```python
for i, block in enumerate(self.blocks):
    if i % 2 == 0:
        x = checkpoint(block, x, use_reentrant=False)  # [B, T, C]
    else:
        x = block(x)  # [B, T, C]
```

Tradeoff:

- lower memory
- more compute
- slower step time

## 10. Debug a Diverging Training Loop

Divergence means loss explodes, becomes `NaN`, or does not recover.

Add checks for finiteness:

```python
def check_finite_tensor(name, x):
    # x: any tensor
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has non-finite values")
```

Training loop with debug hooks:

```python
for step, batch in enumerate(loader):
    input_ids = batch["input_ids"].cuda()  # [B, T]
    labels = batch["labels"].cuda()        # [B, T]

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)  # [B, T, Vocab]
        check_finite_tensor("logits", logits)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
            labels.reshape(-1),                   # [B*T]
        )

    if not torch.isfinite(loss):
        print("bad loss at step", step)
        print("input min/max", input_ids.min().item(), input_ids.max().item())
        save_debug_batch(batch)
        break

    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if not torch.isfinite(grad_norm):
        print("bad grad norm at step", step)
        save_debug_batch(batch)
        break

    optimizer.step()
```

Check gradients by parameter:

```python
for name, p in model.named_parameters():
    if p.grad is None:
        continue

    # p.grad: same shape as p
    if not torch.isfinite(p.grad).all():
        print("non-finite gradient:", name, p.grad.shape)
        break
```

Common causes:

- learning rate too high
- warmup too short
- bad data batch
- labels out of range
- wrong loss mask
- FP16 overflow
- missing gradient clipping
- optimizer state bug
- distributed all-reduce issue

Debug checklist:

```text
1. Reproduce with fixed seed and same batch.
2. Run one batch in FP32/BF16 without distributed training.
3. Check labels are in [0, vocab_size).
4. Check padding labels use ignore_index if needed.
5. Log grad norm and activation norm.
6. Lower LR by 10x.
7. Disable fused/custom kernels if suspicious.
8. Compare single GPU vs distributed.
```

Label sanity:

```python
# labels: [B, T]
assert labels.min() >= 0 or ignore_index in labels
assert labels[labels != ignore_index].max() < vocab_size
```

Commentary: Divergence debugging is about isolating the first bad step. Do not only inspect after the model is already full of `NaN`s.

## Final Training-Coding Checklist

You should now be able to implement and explain:

- DDP training loop
- gradient accumulation
- BF16/FP16 mixed precision
- cosine scheduler with warmup
- AdamW from scratch
- global gradient clipping
- checkpoint save/load
- EMA weights
- activation checkpointing
- divergence debugging instrumentation

For interviews, always explain tensor shapes, where synchronization happens, where memory is saved, and what failure mode each tool is meant to prevent.
