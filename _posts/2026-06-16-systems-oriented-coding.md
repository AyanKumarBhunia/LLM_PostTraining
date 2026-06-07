---
title: "Coding Exercises: Systems-Oriented Implementations"
date: 2026-06-16 12:00:00 +0000
series_order: 10
categories: [interview-prep]
tags: [llm, pytorch, systems, profiling, cuda, inference, kv-cache, distributed-inference]
pin: false
math: false
permalink: /posts/systems-oriented-coding/
---

## Goal

This tutorial covers systems-oriented coding exercises for frontier-model Training Team interviews.

We will implement and discuss:

1. Profile GPU utilization bottlenecks
2. Optimize a slow dataloader
3. Find memory leaks in training
4. Debug CUDA OOM
5. Reduce communication overhead
6. Analyze tensor-parallel communication
7. Optimize inference throughput
8. Implement a KV cache benchmark
9. Measure FLOPs and memory bandwidth
10. Build a minimal distributed inference server

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width
- `H` = number of attention heads
- `D` = head dimension
- `L` = number of transformer layers
- `Vocab` = vocabulary size
- `world_size` = number of distributed processes
- `tp_size` = tensor-parallel group size

## 1. Profile GPU Utilization Bottlenecks

Low GPU utilization means the GPU is waiting instead of doing useful work. In LLM training, common causes are:

- slow dataloader
- CPU preprocessing bottleneck
- host-to-device copy overhead
- too-small batch size
- frequent synchronization
- inefficient kernels
- communication stalls
- memory bandwidth bottlenecks

Start with timing each part of the step.

```python
import time
import torch
import torch.nn.functional as F

def timed_train_step(model, optimizer, batch, device):
    start = time.perf_counter()

    input_ids = batch["input_ids"].to(device, non_blocking=True)  # [B, T]
    labels = batch["labels"].to(device, non_blocking=True)        # [B, T]
    torch.cuda.synchronize()
    after_h2d = time.perf_counter()

    logits = model(input_ids)  # [B, T, Vocab]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
        labels.reshape(-1),                   # [B*T]
    )
    torch.cuda.synchronize()
    after_forward = time.perf_counter()

    loss.backward()
    torch.cuda.synchronize()
    after_backward = time.perf_counter()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    after_step = time.perf_counter()

    return {
        "h2d_ms": 1000 * (after_h2d - start),
        "forward_ms": 1000 * (after_forward - after_h2d),
        "backward_ms": 1000 * (after_backward - after_forward),
        "optimizer_ms": 1000 * (after_step - after_backward),
        "loss": float(loss.detach().cpu()),
    }
```

This is simple but intrusive because `torch.cuda.synchronize()` blocks the CPU. For more accurate GPU timing, use CUDA events.

```python
def cuda_event_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)  # scalar milliseconds
    return out, elapsed_ms
```

PyTorch profiler gives richer operator-level information.

```python
from torch.profiler import profile, ProfilerActivity, record_function

def profile_steps(model, optimizer, loader, device, num_steps=5):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step, batch in zip(range(num_steps), loader):
            with record_function("train_step"):
                input_ids = batch["input_ids"].to(device, non_blocking=True)  # [B, T]
                labels = batch["labels"].to(device, non_blocking=True)        # [B, T]

                logits = model(input_ids)  # [B, T, Vocab]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
                    labels.reshape(-1),                   # [B*T]
                )

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

What to look for:

- `aten::copy_` or `cudaMemcpyAsync` dominating means data movement is expensive.
- Long CPU time before kernels means dataloader or Python overhead.
- Many tiny kernels means launch overhead and poor fusion.
- `ncclAllReduce` dominating means distributed communication is the bottleneck.
- High memory time but low compute utilization means memory bandwidth bottleneck.

Interview answer:

> I would profile the training step into dataloading, host-to-device copy, forward, backward, optimizer, and distributed communication. I would use CUDA events for lightweight timing and PyTorch profiler or Nsight for operator-level analysis. Then I would fix the largest measured bottleneck rather than guessing.

Commentary: GPU utilization alone can be misleading. A job can show high utilization while still being memory-bound or communication-bound.

## 2. Optimize a Slow Dataloader

A slow dataloader starves the GPU. Symptoms include:

- GPU utilization drops between steps
- CPU cores are busy while GPU waits
- training step time varies a lot
- profiler shows gaps before forward pass

A baseline dataloader:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=B,
    shuffle=True,
    num_workers=0,
)
```

This is often too slow because all loading and preprocessing happen in the main process.

Better default:

```python
loader = DataLoader(
    dataset,
    batch_size=B,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

Pinned memory helps asynchronous host-to-device transfer.

```python
for batch in loader:
    input_ids = batch["input_ids"].to("cuda", non_blocking=True)  # [B, T]
    labels = batch["labels"].to("cuda", non_blocking=True)        # [B, T]
```

A common bottleneck is tokenizing text inside `__getitem__`.

```python
class SlowTextDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        text = self.texts[idx]                 # string
        tokens = tokenizer(text).input_ids     # [T], expensive CPU work
        return torch.tensor(tokens)            # [T]
```

Prefer pre-tokenized or packed data.

```python
class PackedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens      # [total_tokens]
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.tokens[start:end]          # [T]
        labels = self.tokens[start + 1:end + 1]     # [T]
        return {"input_ids": input_ids, "labels": labels}
```

Measure dataloader wait time separately.

```python
def measure_loader_wait(loader, num_batches=100):
    iterator = iter(loader)
    wait_times = []

    for _ in range(num_batches):
        start = time.perf_counter()
        batch = next(iterator)
        end = time.perf_counter()

        wait_times.append(end - start)

    return sum(wait_times) / len(wait_times)  # scalar seconds
```

For variable-length examples, a bad collator can waste memory and compute by padding to the longest sequence globally. Pad only within the batch.

```python
def collate_pad(batch, pad_token_id):
    lengths = torch.tensor([len(x["input_ids"]) for x in batch])  # [B]
    max_len = int(lengths.max())

    input_ids = torch.full((len(batch), max_len), pad_token_id)  # [B, T_batch]
    labels = torch.full((len(batch), max_len), -100)             # [B, T_batch]

    for i, example in enumerate(batch):
        ids = example["input_ids"]       # [T_i]
        input_ids[i, :len(ids)] = ids    # [T_i]
        labels[i, :len(ids)] = ids       # [T_i]

    return {"input_ids": input_ids, "labels": labels}
```

Interview answer:

> I would first measure whether the GPU is waiting on data. Then I would use multiple workers, pinned memory, persistent workers, asynchronous GPU copies, prefetching, pre-tokenized data, and efficient batching or packing. If preprocessing is expensive, I would move it offline.

Commentary: More workers are not always better. Too many workers can increase CPU contention, memory use, or filesystem pressure.

## 3. Find Memory Leaks in Training

A training memory leak means GPU memory grows over steps when it should stabilize.

Common causes:

- storing tensors that still require gradients
- appending losses without `.detach()`
- keeping outputs for logging
- accumulating computation graphs
- not clearing references after evaluation
- accidentally using `retain_graph=True`
- variable sequence length causing allocator growth mistaken for a leak

Bad logging example:

```python
losses = []

for batch in loader:
    logits = model(batch["input_ids"].cuda())  # [B, T, Vocab]
    loss = compute_loss(logits, batch["labels"].cuda())  # scalar
    losses.append(loss)  # Bad: keeps computation graph alive
```

Correct logging:

```python
losses = []

for batch in loader:
    logits = model(batch["input_ids"].cuda())  # [B, T, Vocab]
    loss = compute_loss(logits, batch["labels"].cuda())  # scalar
    losses.append(float(loss.detach().cpu()))  # scalar Python float
```

Track memory over time.

```python
def log_cuda_memory(step):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(
        f"step={step} "
        f"allocated={allocated:.2f}GB "
        f"reserved={reserved:.2f}GB "
        f"peak={peak:.2f}GB"
    )
```

Use it during training:

```python
torch.cuda.reset_peak_memory_stats()

for step, batch in enumerate(loader):
    input_ids = batch["input_ids"].cuda(non_blocking=True)  # [B, T]
    labels = batch["labels"].cuda(non_blocking=True)        # [B, T]

    logits = model(input_ids)  # [B, T, Vocab]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
        labels.reshape(-1),                   # [B*T]
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 100 == 0:
        log_cuda_memory(step)
```

For suspicious Python references, inspect live tensors.

```python
import gc

def count_live_cuda_tensors():
    count = 0
    total_bytes = 0

    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            count += 1
            total_bytes += obj.numel() * obj.element_size()

    return count, total_bytes / 1024**3
```

A useful debugging trick is to bisect the training loop:

```python
def memory_after_forward(model, batch):
    input_ids = batch["input_ids"].cuda()  # [B, T]
    logits = model(input_ids)              # [B, T, Vocab]
    return torch.cuda.memory_allocated()

def memory_after_backward(model, batch):
    input_ids = batch["input_ids"].cuda()  # [B, T]
    labels = batch["labels"].cuda()        # [B, T]

    logits = model(input_ids)              # [B, T, Vocab]
    loss = compute_loss(logits, labels)    # scalar
    loss.backward()
    return torch.cuda.memory_allocated()
```

Interview answer:

> I would check whether allocated memory grows step after step, then look for tensors being stored with attached computation graphs. I would detach logged values, avoid retaining outputs, inspect live CUDA tensors, and bisect the loop to find whether the leak happens in forward, backward, eval, or logging.

Commentary: PyTorch's caching allocator keeps reserved memory, so `memory_reserved()` may stay high even when there is no leak. `memory_allocated()` is usually the better signal.

## 4. Debug CUDA OOM

CUDA OOM means the requested allocation could not fit in available GPU memory.

First, read the error carefully:

```text
CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 0 has 80.00 GiB total capacity.
74.00 GiB already allocated.
```

Then identify what consumes memory:

- parameters
- gradients
- optimizer states
- activations
- attention scores
- KV cache
- temporary buffers
- fragmentation

Approximate training memory:

```python
def estimate_param_memory(num_params, bytes_per_param, optimizer_multiplier):
    # num_params: scalar
    # bytes_per_param: scalar, e.g. 2 for bf16
    # optimizer_multiplier: scalar, e.g. params + grads + Adam states
    return num_params * bytes_per_param * optimizer_multiplier
```

Attention memory can explode with sequence length.

```python
def attention_score_memory_gb(B, H, T, bytes_per_value=2):
    # scores: [B, H, T, T]
    num_values = B * H * T * T
    return num_values * bytes_per_value / 1024**3
```

Debug by printing shapes before the OOM point.

```python
def attention_debug(q, k, v):
    # q, k, v: [B, H, T, D]
    print("q", tuple(q.shape))
    print("k", tuple(k.shape))
    print("v", tuple(v.shape))

    scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
    print("scores", tuple(scores.shape))

    weights = torch.softmax(scores / q.size(-1) ** 0.5, dim=-1)  # [B, H, T, T]
    out = weights @ v  # [B, H, T, D]
    return out
```

Common fixes:

- reduce batch size
- reduce sequence length
- use gradient accumulation
- use activation checkpointing
- use mixed precision
- use FlashAttention
- use ZeRO/FSDP
- use smaller optimizer states
- free unused tensors
- reduce KV cache size during inference

Example gradient accumulation to reduce microbatch memory:

```python
accum_steps = 8
optimizer.zero_grad(set_to_none=True)

for micro_step in range(accum_steps):
    input_ids = micro_batches[micro_step]["input_ids"].cuda()  # [micro_B, T]
    labels = micro_batches[micro_step]["labels"].cuda()        # [micro_B, T]

    logits = model(input_ids)  # [micro_B, T, Vocab]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),  # [micro_B*T, Vocab]
        labels.reshape(-1),                   # [micro_B*T]
    )
    loss = loss / accum_steps
    loss.backward()

optimizer.step()
```

Fragmentation can sometimes be helped by allocator settings, but algorithmic memory reduction is usually more important.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Interview answer:

> I would identify whether the OOM comes from parameters, activations, attention, optimizer state, or KV cache. Then I would reduce the dominant term: smaller microbatch, shorter sequence, checkpointing, mixed precision, FlashAttention, sharding, or more efficient cache allocation. I would inspect tensor shapes and memory stats near the failing operation.

Commentary: `torch.cuda.empty_cache()` usually does not fix a real OOM. It may release cached blocks to the driver, but it does not reduce live tensor memory.

## 5. Reduce Communication Overhead

Distributed training and inference often spend time communicating tensors between GPUs.

Common communication operations:

- all-reduce gradients in DDP
- all-gather parameters in FSDP
- reduce-scatter gradients in ZeRO
- all-reduce activations in tensor parallelism
- all-gather logits or hidden states

Measure communication explicitly.

```python
import torch.distributed as dist

def timed_all_reduce(x):
    # x: [N]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)  # scalar milliseconds
```

Reduce communication volume by using gradient accumulation:

```python
for micro_step in range(accum_steps):
    should_sync = micro_step == accum_steps - 1
    context = model.no_sync() if not should_sync else nullcontext()

    with context:
        input_ids = micro_batches[micro_step]["input_ids"].cuda()  # [micro_B, T]
        labels = micro_batches[micro_step]["labels"].cuda()        # [micro_B, T]
        logits = model(input_ids)                                  # [micro_B, T, Vocab]
        loss = compute_loss(logits, labels) / accum_steps          # scalar
        loss.backward()
```

This avoids all-reducing gradients after every microbatch.

Overlap communication with computation when possible. DDP does this with gradient buckets: as soon as a bucket of gradients is ready, it can begin all-reduce while earlier layers are still backpropagating.

```python
ddp_model = torch.nn.parallel.DistributedDataParallel(
    model,
    bucket_cap_mb=25,
    gradient_as_bucket_view=True,
)
```

Other techniques:

- use faster interconnects and topology-aware placement
- keep tensor-parallel groups within NVLink islands
- increase compute per communication with larger microbatches
- fuse small collectives
- avoid unnecessary `.item()` and synchronization
- compress gradients only when accuracy impact is acceptable
- use sequence parallelism to reduce activation memory and balance communication

Interview answer:

> I would first measure which collective dominates, then reduce frequency, reduce bytes, or overlap communication with compute. For training, gradient accumulation, DDP bucket tuning, ZeRO/FSDP choices, and topology-aware process groups matter. For inference, I would minimize tensor-parallel collectives and batch communication efficiently.

Commentary: Communication optimization depends heavily on hardware topology. A strategy that works on 8 GPUs with NVLink may be poor across multiple nodes.

## 6. Analyze Tensor-Parallel Communication

Tensor parallelism splits individual matrix multiplications across GPUs.

Consider a linear layer:

```python
# x: [B, T, C]
# W: [C, 4*C]
y = x @ W  # [B, T, 4*C]
```

Column parallelism splits output features.

```python
# On each rank:
# W_local: [C, 4*C/tp_size]
y_local = x @ W_local  # [B, T, 4*C/tp_size]
```

If the next operation can consume sharded output, no immediate all-gather is needed. If it needs full `y`, then all-gather is required.

```python
def column_parallel_linear(x, weight_local):
    # x: [B, T, C]
    # weight_local: [C, C_out/tp_size]
    y_local = x @ weight_local  # [B, T, C_out/tp_size]
    return y_local              # sharded along hidden/output dimension
```

Row parallelism splits input features.

```python
def row_parallel_linear(x_local, weight_local):
    # x_local: [B, T, C/tp_size]
    # weight_local: [C/tp_size, C_out]
    y_partial = x_local @ weight_local  # [B, T, C_out]

    dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
    return y_partial                    # [B, T, C_out]
```

Transformer MLP often pairs column parallel up-projection with row parallel down-projection.

```python
def tensor_parallel_mlp(x, w_gate_local, w_up_local, w_down_local):
    # x: [B, T, C], replicated on each TP rank
    gate_local = x @ w_gate_local       # [B, T, hidden/tp_size]
    up_local = x @ w_up_local           # [B, T, hidden/tp_size]

    hidden_local = F.silu(gate_local) * up_local  # [B, T, hidden/tp_size]

    out_partial = hidden_local @ w_down_local     # [B, T, C]
    dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)
    return out_partial                            # [B, T, C]
```

Attention also needs careful communication. If heads are split across tensor-parallel ranks, each rank can compute a subset of heads.

```python
def tensor_parallel_attention(x, qkv_local, out_proj_local):
    # x: [B, T, C]
    # qkv_local projects to local heads only
    qkv = qkv_local(x)  # [B, T, 3 * H_local * D]

    q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, H_local * D]
    q = q.view(B, T, H_local, D).transpose(1, 2)  # [B, H_local, T, D]
    k = k.view(B, T, H_local, D).transpose(1, 2)  # [B, H_local, T, D]
    v = v.view(B, T, H_local, D).transpose(1, 2)  # [B, H_local, T, D]

    scores = q @ k.transpose(-2, -1)              # [B, H_local, T, T]
    weights = torch.softmax(scores / D**0.5, dim=-1)  # [B, H_local, T, T]
    context = weights @ v                         # [B, H_local, T, D]

    context = context.transpose(1, 2).reshape(B, T, H_local * D)  # [B, T, C/tp_size]
    out_partial = out_proj_local(context)                         # [B, T, C]

    dist.all_reduce(out_partial, op=dist.ReduceOp.SUM)
    return out_partial                                            # [B, T, C]
```

What to analyze:

- number of collectives per layer
- bytes communicated per collective
- whether collectives are on the critical path
- whether communication overlaps with compute
- whether TP groups match physical topology
- whether sequence parallelism changes the tradeoff

Interview answer:

> I would identify where the tensor-parallel layer uses all-gather, reduce-scatter, or all-reduce, then estimate bytes moved per layer and compare that with compute time. Column-parallel layers shard outputs; row-parallel layers usually need an all-reduce. Good TP design minimizes collectives and keeps them within fast interconnect groups.

Commentary: Tensor-parallel communication formulas vary by implementation. Megatron-style TP, sequence parallelism, expert parallelism, and custom fused kernels can move collectives around.

## 7. Optimize Inference Throughput

Inference throughput is usually measured in tokens per second.

```python
tokens_per_second = total_generated_tokens / elapsed_seconds
```

LLM inference has two phases:

- prefill: process prompt tokens in parallel
- decode: generate one token at a time using KV cache

Naive generation underuses the GPU.

```python
for request in requests:
    input_ids = request.input_ids.unsqueeze(0).cuda()  # [1, T_prompt]
    output = model.generate(input_ids, max_new_tokens=128)  # [1, T_prompt + T_new]
```

Batching improves throughput.

```python
def batch_prefill(model, input_ids, attention_mask):
    # input_ids: [B, T_prompt]
    # attention_mask: [B, T_prompt]
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)  # [B, T_prompt, Vocab]
    return logits[:, -1, :]                                       # [B, Vocab]
```

Continuous batching keeps the GPU full as requests enter and leave.

```python
class DecodeRequest:
    def __init__(self, input_ids, max_new_tokens):
        self.input_ids = input_ids          # [T_prompt]
        self.generated = []                 # list[int]
        self.max_new_tokens = max_new_tokens
        self.done = False

def decode_batch_step(model, active_requests, kv_cache):
    # token_batch: [B, 1], newest token for each active request
    token_batch = torch.stack([
        req.input_ids[-1:] if len(req.generated) == 0 else torch.tensor([req.generated[-1]])
        for req in active_requests
    ]).cuda()

    logits, kv_cache = model.decode_one_token(token_batch, kv_cache)  # logits: [B, 1, Vocab]
    next_tokens = logits[:, -1, :].argmax(dim=-1)                     # [B]

    for req, token in zip(active_requests, next_tokens.tolist()):
        req.generated.append(token)
        req.done = len(req.generated) >= req.max_new_tokens

    return kv_cache
```

Throughput optimizations:

- batch requests together
- use KV caching
- preallocate KV cache
- use paged attention for variable-length requests
- use FlashAttention for prefill
- use fused decode kernels
- use quantized weights when quality allows
- separate prefill and decode scheduling
- avoid Python in the inner decode loop
- tune maximum batch tokens, not only batch size

Measure both latency and throughput.

```python
def benchmark_generation(model, input_ids, max_new_tokens):
    # input_ids: [B, T_prompt]
    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)  # [B, T_prompt + T_new]

    torch.cuda.synchronize()
    end = time.perf_counter()

    generated_tokens = input_ids.size(0) * max_new_tokens
    return {
        "tokens_per_second": generated_tokens / (end - start),
        "latency_seconds": end - start,
        "outputs": outputs,
    }
```

Interview answer:

> I would optimize inference by separating prefill and decode, batching requests, using KV cache, preallocating cache memory, and measuring tokens/sec plus tail latency. For decode-heavy workloads, scheduling and KV-cache memory bandwidth dominate; for prefill-heavy workloads, large matrix multiplies and attention kernels matter more.

Commentary: Maximum throughput and lowest latency are different goals. Large batches improve throughput but can hurt per-request latency.

## 8. Implement a KV Cache Benchmark

A KV cache benchmark should measure:

- allocation cost
- append/write cost
- read bandwidth during attention
- memory footprint
- effect of batch size
- effect of context length
- effect of dtype

Avoid benchmarking repeated `torch.cat` as if it were production KV cache. Production systems usually preallocate.

```python
class StaticKVCache:
    def __init__(self, num_layers, B, H_kv, max_T, D, dtype, device):
        self.k = [
            torch.empty(B, H_kv, max_T, D, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]  # each [B, H_kv, max_T, D]
        self.v = [
            torch.empty(B, H_kv, max_T, D, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]  # each [B, H_kv, max_T, D]
        self.pos = 0

    def append(self, layer_idx, k_new, v_new):
        # k_new, v_new: [B, H_kv, 1, D]
        self.k[layer_idx][:, :, self.pos:self.pos + 1, :] = k_new
        self.v[layer_idx][:, :, self.pos:self.pos + 1, :] = v_new

    def advance(self):
        self.pos += 1
```

Benchmark writes.

```python
def benchmark_kv_writes(num_layers, B, H_kv, max_T, D, dtype=torch.bfloat16):
    cache = StaticKVCache(num_layers, B, H_kv, max_T, D, dtype, "cuda")

    k_new = torch.randn(B, H_kv, 1, D, dtype=dtype, device="cuda")  # [B, H_kv, 1, D]
    v_new = torch.randn(B, H_kv, 1, D, dtype=dtype, device="cuda")  # [B, H_kv, 1, D]

    torch.cuda.synchronize()
    start = time.perf_counter()

    for t in range(max_T):
        for layer in range(num_layers):
            cache.append(layer, k_new, v_new)
        cache.advance()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_written = 2 * num_layers * B * H_kv * max_T * D * torch.tensor([], dtype=dtype).element_size()
    bandwidth_gb_s = bytes_written / elapsed / 1024**3
    return bandwidth_gb_s
```

Benchmark attention reads.

```python
def benchmark_kv_attention_read(B, H_q, H_kv, T_cache, D, dtype=torch.bfloat16):
    q = torch.randn(B, H_q, 1, D, dtype=dtype, device="cuda")          # [B, H_q, 1, D]
    k = torch.randn(B, H_kv, T_cache, D, dtype=dtype, device="cuda")   # [B, H_kv, T_cache, D]
    v = torch.randn(B, H_kv, T_cache, D, dtype=dtype, device="cuda")   # [B, H_kv, T_cache, D]

    if H_q != H_kv:
        repeat = H_q // H_kv
        k = k.repeat_interleave(repeat, dim=1)  # [B, H_q, T_cache, D]
        v = v.repeat_interleave(repeat, dim=1)  # [B, H_q, T_cache, D]

    torch.cuda.synchronize()
    start = time.perf_counter()

    scores = q @ k.transpose(-2, -1)                  # [B, H_q, 1, T_cache]
    weights = torch.softmax(scores / D**0.5, dim=-1)  # [B, H_q, 1, T_cache]
    out = weights @ v                                 # [B, H_q, 1, D]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_read = 2 * B * H_q * T_cache * D * q.element_size()
    return out, bytes_read / elapsed / 1024**3
```

Sweep context lengths.

```python
for T_cache in [1024, 4096, 8192, 32768, 131072]:
    _, bandwidth = benchmark_kv_attention_read(
        B=8,
        H_q=32,
        H_kv=8,
        T_cache=T_cache,
        D=128,
    )
    print(T_cache, bandwidth)
```

Interview answer:

> I would benchmark KV cache with preallocated memory, measuring write cost during decode and read cost during attention across batch size, context length, head count, dtype, and cache layout. I would avoid `torch.cat` in the benchmark unless I specifically wanted to show why dynamic concatenation is slow.

Commentary: The simple benchmark above materializes repeated GQA heads, which production kernels often avoid. It is useful for explanation but not a perfect measurement of optimized serving kernels.

## 9. Measure FLOPs and Memory Bandwidth

Performance depends on both compute and memory movement.

Approximate linear-layer FLOPs:

```python
def linear_flops(B, T, C_in, C_out):
    # y = x @ W, x: [B, T, C_in], W: [C_in, C_out]
    return 2 * B * T * C_in * C_out
```

The factor `2` counts multiply and add.

Approximate attention FLOPs:

```python
def attention_flops(B, H, T, D):
    qk = 2 * B * H * T * T * D      # q @ k^T
    av = 2 * B * H * T * T * D      # attn @ v
    return qk + av
```

Approximate bytes for a tensor:

```python
def tensor_bytes(shape, dtype):
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * element_size
```

Measure achieved TFLOPs for an operation.

```python
def benchmark_matmul(B, T, C_in, C_out, dtype=torch.bfloat16, iters=100):
    x = torch.randn(B, T, C_in, device="cuda", dtype=dtype)       # [B, T, C_in]
    w = torch.randn(C_in, C_out, device="cuda", dtype=dtype)      # [C_in, C_out]

    for _ in range(10):
        y = x @ w  # [B, T, C_out]

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        y = x @ w  # [B, T, C_out]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    flops = linear_flops(B, T, C_in, C_out) * iters
    tflops = flops / elapsed / 1e12
    return y, tflops
```

Measure memory bandwidth with a copy-like operation.

```python
def benchmark_memory_bandwidth(num_elements, dtype=torch.bfloat16, iters=100):
    x = torch.randn(num_elements, device="cuda", dtype=dtype)  # [N]
    y = torch.empty_like(x)                                    # [N]

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        y.copy_(x)  # [N]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bytes_moved = 2 * x.numel() * x.element_size() * iters
    gb_s = bytes_moved / elapsed / 1024**3
    return gb_s
```

Arithmetic intensity helps classify a kernel.

```python
def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved  # FLOPs per byte
```

High arithmetic intensity usually means compute-bound. Low arithmetic intensity usually means memory-bandwidth-bound.

Example:

```python
B, T, C = 8, 2048, 4096
flops = linear_flops(B, T, C, 4 * C)
bytes_moved = tensor_bytes((B, T, C), torch.bfloat16) + tensor_bytes((C, 4 * C), torch.bfloat16)
intensity = arithmetic_intensity(flops, bytes_moved)
```

Interview answer:

> I would estimate FLOPs from the dominant matrix multiplications and attention operations, then measure elapsed GPU time to compute achieved TFLOPs. For memory bandwidth, I would estimate bytes read and written and compare achieved GB/s against hardware peak. Arithmetic intensity helps decide whether an operation is compute-bound or memory-bound.

Commentary: FLOP estimates for full transformer steps are approximate because fused kernels, recomputation, dropout, normalization, optimizer states, and communication complicate the exact count.

## 10. Build a Minimal Distributed Inference Server

A minimal distributed inference server has three parts:

- request handling
- distributed model workers
- response collection

For an interview, a clear toy design is better than a huge production system.

Architecture:

```text
client -> HTTP server -> request queue -> GPU workers -> response queue -> client
```

Each GPU worker owns one process.

```python
def init_worker(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
```

In simple data-parallel inference, every GPU has a full model copy and handles different requests.

```python
class InferenceWorker:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids  # [1, T_prompt]
        input_ids = input_ids.to(self.device)                              # [1, T_prompt]

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )  # [1, T_prompt + T_new]

        return self.tokenizer.decode(output_ids[0])  # string
```

A minimal queue-based server:

```python
import queue
import threading

request_queue = queue.Queue()
response_map = {}

def worker_loop(worker):
    while True:
        request = request_queue.get()
        if request is None:
            break

        request_id = request["id"]
        prompt = request["prompt"]
        max_new_tokens = request["max_new_tokens"]

        text = worker.generate(prompt, max_new_tokens)
        response_map[request_id] = text
```

HTTP handler concept:

```python
def handle_generate(prompt, max_new_tokens):
    request_id = make_request_id()

    request_queue.put({
        "id": request_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    })

    while request_id not in response_map:
        time.sleep(0.001)

    return response_map.pop(request_id)
```

For tensor-parallel inference, each request is processed by a group of GPUs, and layers contain collectives.

```python
def tensor_parallel_generate(model, input_ids, max_new_tokens):
    # input_ids: [B, T_prompt], replicated across TP ranks
    tokens = input_ids  # [B, current_T]
    kv_cache = model.allocate_kv_cache(batch_size=input_ids.size(0))

    for _ in range(max_new_tokens):
        logits, kv_cache = model.decode(tokens[:, -1:], kv_cache)  # logits: [B, 1, Vocab/tp_size] or [B, 1, Vocab]

        # If logits are sharded over vocabulary, gather or compute distributed argmax.
        full_logits = gather_vocab_logits_if_needed(logits)        # [B, 1, Vocab]
        next_token = full_logits[:, -1, :].argmax(dim=-1)          # [B]

        tokens = torch.cat([tokens, next_token[:, None]], dim=-1)  # [B, current_T + 1]

    return tokens                                                  # [B, T_prompt + T_new]
```

Production systems add many missing pieces:

- continuous batching
- cancellation
- timeouts
- streaming tokens
- authentication
- request prioritization
- paged KV cache
- health checks
- metrics
- autoscaling
- fault tolerance

Minimal metrics:

```python
class ServerMetrics:
    def __init__(self):
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    def record(self, latency, generated_tokens):
        self.total_requests += 1
        self.total_tokens += generated_tokens
        self.total_latency += latency

    def tokens_per_second(self):
        return self.total_tokens / max(self.total_latency, 1e-6)
```

Interview answer:

> I would build a minimal server with an HTTP front end, request queue, one GPU worker per process, model replicas or tensor-parallel groups, and a response path. For a toy implementation, data-parallel workers are simplest. For large models, tensor parallelism requires distributed collectives during each forward pass and a scheduler that batches requests efficiently.

Commentary: A real distributed inference server is a systems project, not just a PyTorch script. The minimal version is useful for explaining architecture, but production quality requires scheduling, streaming, observability, and failure handling.
