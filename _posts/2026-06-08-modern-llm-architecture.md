---
title: "Section B: Modern LLM Architecture"
date: 2026-06-08 12:00:00 +0000
series_order: 2
categories: [interview-prep]
tags: [llm, architecture, rmsnorm, swiglu, moe, gqa, llama, gpt]
pin: false
math: false
permalink: /posts/modern-llm-architecture/
---

## Goal

This is the second section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover modern decoder-only LLM architecture:

1. Why RMSNorm is preferred over LayerNorm in many LLMs
2. PreNorm vs PostNorm
3. Why SwiGLU is commonly used
4. Why MLP layers are important
5. Why transformer blocks are so deep
6. Residual connections
7. Residual stream interference
8. Mixture-of-Experts models
9. MoE advantages over dense models
10. Expert routing challenges
11. Load balancing loss
12. Grouped query attention
13. Why GQA helps inference
14. Multi-query attention
15. Why Llama architecture differs from GPT-2

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width, also called `d_model`
- `H_q` = number of query heads
- `H_kv` = number of key/value heads
- `D` = head dimension
- `E` = number of experts
- `K` = number of selected experts per token

## 16. Why Is RMSNorm Preferred Over LayerNorm in Many LLMs?

LayerNorm normalizes by subtracting the mean and dividing by the standard deviation.

```python
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., C]
        mean = x.mean(dim=-1, keepdim=True)  # [..., 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [..., 1]
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # [..., C]
        return self.weight * x_hat + self.bias  # [..., C]
```

RMSNorm does not subtract the mean. It only rescales by the root mean square.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [..., C]
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # [..., 1]
        return self.weight * x * rms  # [..., C]
```

Why many LLMs prefer RMSNorm:

- It is cheaper: no mean subtraction and often no bias.
- It is simple and stable in large PreNorm transformers.
- It preserves the direction of the residual stream while controlling scale.
- Empirically it works well in Llama-style models.

Interview answer:

> RMSNorm is preferred because it provides most of the stabilizing effect of LayerNorm while being simpler and cheaper. In modern PreNorm decoder-only LLMs, controlling activation scale is often enough; mean-centering is not always necessary.

Commentary: This is partly empirical. RMSNorm is not universally "better" in all architectures, but it is a strong default in many modern LLMs.

## 17. What Is PreNorm vs PostNorm?

A transformer block has attention, MLP, residual connections, and normalization.

The difference is where the normalization happens.

### PostNorm

Original Transformer-style blocks often used PostNorm:

```python
def postnorm_block(x):
    # x: [B, T, C]
    x = norm1(x + attention(x))  # [B, T, C]
    x = norm2(x + mlp(x))        # [B, T, C]
    return x
```

PostNorm normalizes after the residual addition.

### PreNorm

Most modern LLMs use PreNorm:

```python
def prenorm_block(x):
    # x: [B, T, C]
    x = x + attention(norm1(x))  # [B, T, C]
    x = x + mlp(norm2(x))        # [B, T, C]
    return x
```

PreNorm normalizes before each sublayer.

Why PreNorm helps deep LLMs:

- The residual path has a cleaner gradient route.
- Deep networks are less likely to become unstable.
- It is easier to scale to many layers.

Tradeoff:

- PreNorm may leave the residual stream less normalized after the block.
- Many models add a final normalization before the language-model head.

Modern decoder block:

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, attn, mlp):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.attn_norm(x))  # [B, T, C]
        x = x + self.mlp(self.mlp_norm(x))    # [B, T, C]
        return x                              # [B, T, C]
```

## 18. Why Is SwiGLU Commonly Used?

SwiGLU is a gated MLP activation. It is common in modern LLMs because it tends to improve quality compared with older ReLU/GELU MLPs at similar compute.

Standard MLP:

```python
class GELUMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        h = F.gelu(self.up(x))  # [B, T, hidden_dim]
        return self.down(h)    # [B, T, C]
```

SwiGLU:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, C]
        gate = F.silu(self.gate(x))  # [B, T, hidden_dim]
        up = self.up(x)              # [B, T, hidden_dim]
        h = gate * up                # [B, T, hidden_dim]
        return self.down(h)          # [B, T, C]
```

Why it helps:

- The gate controls which features pass through.
- Multiplicative interactions increase expressiveness.
- Empirically strong in large language models.

Interview answer:

> SwiGLU is used because gated MLPs are more expressive than plain activations and have shown strong empirical performance in modern LLMs.

## 19. Why Are MLP Layers Important?

Attention mixes information across tokens. MLP layers transform information within each token position.

Attention answers:

> Which other tokens should this token read from?

MLP answers:

> After gathering context, how should this token's features be transformed?

In a decoder block:

```python
def block(x):
    # x: [B, T, C]
    x = x + attention(norm1(x))  # token mixing: [B, T, C]
    x = x + mlp(norm2(x))        # per-token feature transform: [B, T, C]
    return x
```

An MLP is applied independently at each position:

```python
# x: [B, T, C]
h = up_proj(x)       # [B, T, hidden_dim]
h = activation(h)    # [B, T, hidden_dim]
y = down_proj(h)     # [B, T, C]
```

Why MLPs matter:

- They store a large fraction of model parameters.
- They provide nonlinear feature transformations.
- They can act like key-value memory over learned features.
- They complement attention: attention moves information, MLPs process it.

Commentary: Some mechanistic-interpretability work views MLP neurons as feature detectors or memory-like components. That framing is useful, but not the whole story.

## 20. Why Are Transformer Blocks So Deep?

Depth allows repeated stages of communication and computation.

One layer can do:

```text
attend -> transform
```

Many layers can do:

```text
attend -> transform -> attend -> transform -> ... -> answer
```

Why depth helps:

- Later layers can build on features computed by earlier layers.
- Multi-hop reasoning may require several rounds of information flow.
- Different layers can specialize in different abstraction levels.
- More layers increase effective computation per token.

Shape stays the same across most blocks:

```python
x = token_embedding(input_ids)  # [B, T, C]

for block in blocks:
    x = block(x)                # [B, T, C]

logits = lm_head(final_norm(x)) # [B, T, Vocab]
```

Interview answer:

> Transformer blocks are deep because language modeling requires many rounds of token mixing and nonlinear feature transformation. Depth gives the model iterative computation while the residual stream preserves information across layers.

## 21. Explain Residual Connections

A residual connection adds the input of a sublayer back to its output:

```python
y = x + sublayer(x)
```

In a modern LLM block:

```python
def forward(x):
    # x: [B, T, C]
    attn_out = attention(norm1(x))  # [B, T, C]
    x = x + attn_out                # [B, T, C]

    mlp_out = mlp(norm2(x))         # [B, T, C]
    x = x + mlp_out                 # [B, T, C]
    return x
```

Why residuals are important:

- They improve gradient flow.
- They let layers make incremental updates instead of rewriting everything.
- They preserve information across many layers.
- They make very deep networks easier to optimize.

The residual stream is the running representation that all layers read from and write to.

```python
# x is the residual stream: [B, T, C]
for layer in layers:
    x = x + layer_update(x)  # [B, T, C]
```

## 22. What Causes Residual Stream Interference?

The residual stream is shared. Attention and MLP sublayers across all layers write updates into the same vector space.

Residual stream interference happens when multiple features or layer updates compete inside the same limited-dimensional representation.

Possible causes:

- Too many features packed into the same dimensions
- Large updates overwriting useful information
- Poorly scaled residual branches
- Conflicting objectives in different parts of training data
- Attention and MLP outputs pushing the stream in incompatible directions

Simple toy view:

```python
# x: [B, T, C], shared residual stream
update_a = feature_writer_a(x)  # [B, T, C]
update_b = feature_writer_b(x)  # [B, T, C]

x = x + update_a + update_b     # [B, T, C]
```

If `update_a` and `update_b` use the same dimensions for different meanings, later layers may have trouble reading the intended feature cleanly.

Why normalization and residual scaling matter:

```python
# x: [B, T, C]
update = sublayer(norm(x))  # [B, T, C]
x = x + update              # [B, T, C]
```

Normalization helps control scale before each update. Some architectures also use initialization tricks or residual scaling to prevent updates from overwhelming the stream.

Commentary: "Residual stream interference" is not one single precisely defined failure mode. In interviews, use it to mean competition or superposition of features in the shared residual representation.

## 23. What Are Mixture-of-Experts Models?

Mixture-of-Experts, or MoE, models replace some dense layers with multiple expert networks.

Instead of every token going through the same MLP, a router chooses which expert MLPs process each token.

Dense MLP:

```python
# x: [B, T, C]
y = dense_mlp(x)  # [B, T, C]
```

MoE MLP:

```python
# x: [B, T, C]
router_logits = router(x)  # [B, T, E]
top_values, top_indices = torch.topk(router_logits, k=2, dim=-1)
# top_values, top_indices: [B, T, K]
```

Toy MoE implementation:

```python
class ToyMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, n_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(d_model, hidden_dim)
            for _ in range(n_experts)
        ])
        self.top_k = top_k

    def forward(self, x):
        # x: [B, T, C]
        router_logits = self.router(x)  # [B, T, E]
        weights, indices = torch.topk(router_logits, self.top_k, dim=-1)
        # weights, indices: [B, T, K]

        weights = torch.softmax(weights, dim=-1)  # [B, T, K]
        out = torch.zeros_like(x)                 # [B, T, C]

        for slot in range(self.top_k):
            expert_id = indices[..., slot]             # [B, T]
            expert_weight = weights[..., slot][..., None]  # [B, T, 1]

            for i, expert in enumerate(self.experts):
                mask = expert_id == i  # [B, T]
                if mask.any():
                    expert_in = x[mask]       # [tokens_for_expert_i, C]
                    expert_out = expert(expert_in)  # [tokens_for_expert_i, C]
                    out[mask] += expert_weight[mask] * expert_out

        return out  # [B, T, C]
```

This code is clear but inefficient. Production MoE uses optimized token dispatch, grouped matmuls, expert parallelism, and capacity management.

## 24. Advantages of MoE Over Dense Models

Dense models activate all parameters for every token.

MoE models have many parameters but activate only a subset per token.

Advantages:

- More total parameters without proportional compute increase
- Conditional computation: different tokens can use different experts
- Better scaling when training compute is constrained
- Expert specialization may emerge

Example:

```text
Dense model:
  100B total parameters, 100B active per token

MoE model:
  500B total parameters, 50B active per token
```

The MoE model can have much more capacity while keeping active compute closer to a smaller dense model.

Interview answer:

> MoE gives high parameter capacity with sparse activation. It can improve quality per unit of training or inference compute, but it introduces routing, load balancing, and systems complexity.

## 25. Challenges of Expert Routing

Expert routing decides which experts process each token.

Challenges:

- Load imbalance: too many tokens choose the same expert.
- Expert collapse: the router uses only a few experts.
- Routing instability: small changes in logits can change expert assignment.
- Communication cost: tokens must be sent to experts across devices.
- Capacity limits: each expert can process only a fixed number of tokens efficiently.
- Training complexity: router gradients can be noisy.

Example issue:

```python
# router_logits: [B, T, E]
chosen_expert = router_logits.argmax(dim=-1)  # [B, T]

# If most entries equal 0, expert 0 is overloaded.
tokens_per_expert = torch.bincount(chosen_expert.flatten(), minlength=E)  # [E]
```

If one expert receives too many tokens, the system may:

- drop tokens
- reroute tokens
- exceed capacity
- slow down due to stragglers

Commentary: In real MoE systems, routing is both an ML problem and a distributed-systems problem.

## 26. What Is Load Balancing Loss?

Load balancing loss encourages the router to spread tokens across experts.

Without it, the router may send most tokens to a small number of experts.

Simple intuition:

```python
# router_probs: [B, T, E]
router_probs = torch.softmax(router_logits, dim=-1)  # [B, T, E]

# Average probability assigned to each expert.
prob_per_expert = router_probs.mean(dim=(0, 1))  # [E]

# A simple penalty: high if probability mass is concentrated.
loss_balance = E * (prob_per_expert ** 2).sum()
```

If all experts are used equally:

```text
prob_per_expert approx [1/E, 1/E, ..., 1/E]
```

The load balancing loss is lower.

A more realistic version may also consider actual selected tokens:

```python
# selected: [B, T], expert index chosen by top-1 routing
selected = router_logits.argmax(dim=-1)  # [B, T]
one_hot = F.one_hot(selected, num_classes=E).float()  # [B, T, E]

tokens_per_expert = one_hot.mean(dim=(0, 1))  # [E]
prob_per_expert = router_probs.mean(dim=(0, 1))  # [E]

load_balance_loss = E * (tokens_per_expert * prob_per_expert).sum()
```

The exact formula differs across MoE papers and systems, but the goal is the same: avoid expert underuse and overload.

## 27. What Is Grouped Query Attention?

Grouped Query Attention, or GQA, is a compromise between standard multi-head attention and multi-query attention.

Standard multi-head attention:

```text
H_q = H_kv
```

Each query head has its own key and value head.

GQA:

```text
H_q > H_kv
```

Several query heads share one key/value head.

Pseudocode:

```python
# q: [B, H_q, T, D]
# k: [B, H_kv, T, D]
# v: [B, H_kv, T, D]

repeat = H_q // H_kv
k_expanded = k.repeat_interleave(repeat, dim=1)  # [B, H_q, T, D]
v_expanded = v.repeat_interleave(repeat, dim=1)  # [B, H_q, T, D]

scores = q @ k_expanded.transpose(-2, -1)  # [B, H_q, T, T]
weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H_q, T, T]
out = weights @ v_expanded  # [B, H_q, T, D]
```

This simple code physically repeats keys and values. Production kernels avoid unnecessary materialization.

## 28. Why Does GQA Help Inference?

During autoregressive decoding, the model stores keys and values for all previous tokens in a KV cache.

KV cache shape:

```python
# For one layer:
# k_cache, v_cache: [B, H_kv, T_cache, D]
```

The cache size grows with:

```text
B * num_layers * H_kv * T_cache * D
```

Standard MHA has `H_kv = H_q`.

GQA uses fewer KV heads:

```text
H_kv < H_q
```

So the KV cache is smaller.

Example:

```text
H_q = 32
H_kv = 8
```

The KV cache is roughly 4x smaller than standard MHA for the same `D`, sequence length, and number of layers.

Why this matters:

- Decode is often memory-bandwidth-bound.
- Each new token reads from the KV cache.
- Smaller KV cache reduces memory traffic.
- This improves latency and throughput.

Interview answer:

> GQA helps inference by reducing the number of key/value heads stored and read during decoding. This lowers KV-cache memory and memory bandwidth while preserving more quality than pure multi-query attention.

## 29. What Is Multi-Query Attention?

Multi-Query Attention, or MQA, is an extreme version of GQA.

In MQA:

```text
H_kv = 1
```

All query heads share the same key and value head.

```python
# q: [B, H_q, T, D]
# k: [B, 1, T, D]
# v: [B, 1, T, D]

k_shared = k.expand(B, H_q, T, D)  # [B, H_q, T, D], view-style if possible
v_shared = v.expand(B, H_q, T, D)  # [B, H_q, T, D]

scores = q @ k_shared.transpose(-2, -1)  # [B, H_q, T, T]
weights = torch.softmax(scores / math.sqrt(D), dim=-1)  # [B, H_q, T, T]
out = weights @ v_shared  # [B, H_q, T, D]
```

MQA gives maximum KV-cache savings but can hurt model quality more than GQA because all query heads share the same K/V representation.

Relationship:

```text
MHA: H_kv = H_q
GQA: 1 < H_kv < H_q
MQA: H_kv = 1
```

## 30. Why Is Llama Architecture Different From GPT-2?

Llama-style models and GPT-2 are both decoder-only transformers, but many architectural choices changed.

### GPT-2 Style

GPT-2 uses:

- learned absolute positional embeddings
- LayerNorm
- GELU MLP
- multi-head attention
- learned position table added to token embeddings

Simplified GPT-2-style block:

```python
class GPT2StyleBlock(nn.Module):
    def __init__(self, d_model, attn, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.ln1(x))  # [B, T, C]
        x = x + self.mlp(self.ln2(x))   # [B, T, C]
        return x                        # [B, T, C]
```

Note: GPT-2 uses a PreNorm-like block structure with LayerNorm before sublayers.

### Llama Style

Llama-style models use:

- RoPE instead of learned absolute positional embeddings
- RMSNorm instead of LayerNorm
- SwiGLU instead of GELU MLP
- no attention/MLP biases in many projections
- GQA in later/larger versions
- tokenizer and training recipe differences

Simplified Llama-style block:

```python
class LlamaStyleBlock(nn.Module):
    def __init__(self, d_model, attn_with_rope, swiglu):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.attn = attn_with_rope
        self.mlp = swiglu

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.attn_norm(x))  # [B, T, C]
        x = x + self.mlp(self.mlp_norm(x))    # [B, T, C]
        return x                              # [B, T, C]
```

High-level comparison:

```text
GPT-2:
  position: learned absolute embeddings
  norm: LayerNorm
  MLP: GELU
  attention: MHA

Llama-style:
  position: RoPE
  norm: RMSNorm
  MLP: SwiGLU
  attention: MHA or GQA depending on version
```

Why these changes matter:

- RoPE improves relative position handling and context extension behavior.
- RMSNorm is simpler and efficient.
- SwiGLU improves MLP expressiveness.
- GQA reduces inference KV-cache cost.

Commentary: "Llama vs GPT-2" is not just architecture. Data, tokenizer, scale, training compute, optimizer settings, and post-training are also very different. In an interview, separate architectural differences from training-recipe differences.

## Final Interview Checklist

You should now be able to answer all Section B questions:

- RMSNorm controls scale more cheaply than LayerNorm.
- PreNorm normalizes before sublayers and improves deep training stability.
- SwiGLU adds useful gating to MLPs.
- MLPs transform per-token features and store much of the model capacity.
- Deep blocks allow repeated communication and computation.
- Residual connections preserve information and improve gradients.
- Residual stream interference is feature/update competition in the shared stream.
- MoE uses routers to send tokens to expert networks.
- MoE increases total capacity without activating all parameters per token.
- Expert routing has load, stability, and systems challenges.
- Load balancing loss encourages experts to be used evenly.
- GQA shares K/V heads across groups of query heads.
- GQA helps inference by shrinking the KV cache.
- MQA shares one K/V head across all query heads.
- Llama-style models differ from GPT-2 through RoPE, RMSNorm, SwiGLU, and often GQA, along with major training-recipe differences.
