---
title: "Coding Exercises: Research Implementations"
date: 2026-06-17 12:00:00 +0000
series_order: 11
categories: [interview-prep]
tags: [llm, research-coding, scaling-laws, evaluation, rag, speculative-decoding, moe]
pin: false
math: false
permalink: /posts/research-coding/
---

## Goal

This tutorial covers research-coding exercises that often appear in frontier-model Training Team interviews.

We will implement:

1. Chinchilla compute estimation
2. Scaling-law fitting
3. Evaluation harness
4. Perplexity computation
5. Retrieval-augmented generation
6. Speculative decoding
7. RoPE vs ALiBi comparison
8. Tiny transformer training
9. Sparse attention
10. Toy Mixture-of-Experts model

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width
- `H` = number of attention heads
- `D` = head dimension
- `Vocab` = vocabulary size
- `N` = number of examples or parameters depending on context
- `M` = number of retrieval documents
- `E` = number of experts
- `K` = top-k, beam size, or retrieved-document count depending on context

## 1. Implement Chinchilla Compute Estimation

For dense decoder-only transformers, a common back-of-the-envelope training compute estimate is:

```text
training FLOPs ≈ 6 * parameters * tokens
```

Implementation:

```python
def estimate_training_flops(params, tokens):
    # params: scalar, number of model parameters
    # tokens: scalar, number of training tokens
    # returns: scalar FLOPs estimate
    return 6 * params * tokens

def estimate_tokens_for_compute(compute_budget, params):
    # compute_budget: scalar FLOPs
    # params: scalar model parameters
    return compute_budget / (6 * params)

def estimate_params_for_compute(compute_budget, tokens):
    # compute_budget: scalar FLOPs
    # tokens: scalar training tokens
    return compute_budget / (6 * tokens)
```

Sweep candidate model sizes:

```python
compute_budget = 1e24
candidate_params = torch.tensor([3e9, 7e9, 13e9, 30e9, 70e9])  # [num_candidates]

candidate_tokens = compute_budget / (6 * candidate_params)  # [num_candidates]
candidate_flops = estimate_training_flops(candidate_params, candidate_tokens)
# candidate_flops: [num_candidates], approximately compute_budget for each candidate
```

Chinchilla-style planning asks:

> Given fixed compute, which parameter/token pair is expected to give the lowest loss?

Commentary: The `6 * N * D` rule is a rough estimate. Real training FLOPs depend on architecture, sequence length, attention variant, vocabulary head, activation recomputation, hardware efficiency, and implementation details.

## 2. Reproduce a Scaling Law Fit

A simple scaling-law model predicts loss from parameters and tokens.

Toy model:

```python
def predict_loss(log_params, log_tokens, coeffs):
    # log_params: [R], log parameter count for R runs
    # log_tokens: [R], log token count for R runs
    # coeffs: [3] = [bias, alpha, beta]
    # returns: [R] predicted loss
    bias, alpha, beta = coeffs
    return bias + alpha * log_params + beta * log_tokens
```

This is a log-linear toy fit, not the exact Kaplan or Chinchilla form. It is useful for interview coding because it shows the mechanics of fitting and extrapolation.

Fit with least squares:

```python
def fit_log_linear_scaling(params, tokens, losses):
    # params: [R]
    # tokens: [R]
    # losses: [R]
    log_params = torch.log(params)  # [R]
    log_tokens = torch.log(tokens)  # [R]

    ones = torch.ones_like(log_params)  # [R]
    X = torch.stack([ones, log_params, log_tokens], dim=1)  # [R, 3]
    y = losses[:, None]  # [R, 1]

    # Solve min ||X coeffs - y||^2.
    solution = torch.linalg.lstsq(X, y).solution  # [3, 1]
    coeffs = solution[:, 0]  # [3]
    return coeffs
```

Use the fit:

```python
params = torch.tensor([1e8, 3e8, 1e9, 3e9, 7e9])       # [R]
tokens = torch.tensor([5e9, 15e9, 50e9, 150e9, 300e9]) # [R]
losses = torch.tensor([3.2, 2.9, 2.55, 2.32, 2.20])    # [R]

coeffs = fit_log_linear_scaling(params, tokens, losses)  # [3]

future_params = torch.tensor([13e9, 30e9])      # [2]
future_tokens = torch.tensor([500e9, 1000e9])   # [2]
pred = predict_loss(torch.log(future_params), torch.log(future_tokens), coeffs)
# pred: [2]
```

Commentary: Real scaling-law fits often use nonlinear power-law forms with an irreducible loss term. If you are not sure of the exact formula in an interview, say so and implement a clean fitting framework.

## 3. Build an Evaluation Harness

An evaluation harness runs a model on a dataset and computes metrics in a repeatable way.

Define an example format:

```python
examples = [
    {
        "prompt": "Q: 2 + 2 = ?\nA:",
        "answer": "4",
        "task": "math",
    },
]
```

Model interface:

```python
@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    # input_ids: [B=1, T_prompt]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
    )
    # output_ids: [1, T_prompt + T_new]

    generated = output_ids[:, input_ids.size(1):]  # [1, T_new]
    return tokenizer.decode(generated[0], skip_special_tokens=True)
```

Metric:

```python
def exact_match(pred, gold):
    return pred.strip().lower() == gold.strip().lower()
```

Harness:

```python
def run_eval(model, tokenizer, examples):
    results = []

    for ex in examples:
        pred = generate_answer(model, tokenizer, ex["prompt"])
        correct = exact_match(pred, ex["answer"])

        results.append({
            "task": ex["task"],
            "prompt": ex["prompt"],
            "prediction": pred,
            "answer": ex["answer"],
            "correct": correct,
        })

    accuracy = sum(r["correct"] for r in results) / len(results)
    return {"accuracy": accuracy, "results": results}
```

Add per-task aggregation:

```python
def aggregate_by_task(results):
    buckets = {}
    for r in results:
        buckets.setdefault(r["task"], []).append(r["correct"])

    return {
        task: sum(values) / len(values)
        for task, values in buckets.items()
    }
```

What to explain:

- fixed prompts
- deterministic decoding for eval when appropriate
- saved predictions
- per-task metrics
- versioned datasets

## 4. Implement Perplexity Computation

Perplexity is:

```text
exp(mean negative log likelihood)
```

Implementation:

```python
@torch.no_grad()
def compute_perplexity(model, loader, ignore_index=-100):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].cuda()  # [B, T]

        # Predict next token.
        logits = model(input_ids[:, :-1])  # [B, T-1, Vocab]
        targets = input_ids[:, 1:]         # [B, T-1]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*(T-1), Vocab]
            targets.reshape(-1),                  # [B*(T-1)]
            ignore_index=ignore_index,
            reduction="sum",
        )

        valid = (targets != ignore_index).sum().item()
        total_loss += loss.item()
        total_tokens += valid

    mean_nll = total_loss / total_tokens
    return math.exp(mean_nll)
```

Common bugs:

- forgetting the one-token shift
- averaging per batch instead of per token
- not respecting padding or `ignore_index`
- mixing train/eval modes

## 5. Implement Retrieval-Augmented Generation

Retrieval-augmented generation, or RAG, retrieves relevant documents and places them into the prompt.

Toy retriever with dot-product embeddings:

```python
@torch.no_grad()
def retrieve(query_emb, doc_embs, docs, top_k=3):
    # query_emb: [C]
    # doc_embs: [M, C]
    # docs: list length M
    scores = doc_embs @ query_emb  # [M]
    top_scores, top_idx = torch.topk(scores, top_k)  # each [K]
    retrieved_docs = [docs[i] for i in top_idx.tolist()]
    return retrieved_docs, top_scores  # list length K, [K]
```

Build the prompt:

```python
def build_rag_prompt(question, retrieved_docs):
    context = "\n\n".join(
        f"[Document {i+1}]\n{doc}"
        for i, doc in enumerate(retrieved_docs)
    )

    return f"""Use the context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""
```

Generate:

```python
@torch.no_grad()
def rag_answer(model, tokenizer, embedder, question, doc_embs, docs, top_k=3):
    query_emb = embedder(question)  # [C]
    retrieved_docs, scores = retrieve(query_emb, doc_embs, docs, top_k=top_k)

    prompt = build_rag_prompt(question, retrieved_docs)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    # input_ids: [1, T_prompt]

    output_ids = model.generate(input_ids, max_new_tokens=128)
    # output_ids: [1, T_prompt + T_new]

    answer_ids = output_ids[:, input_ids.size(1):]  # [1, T_new]
    return tokenizer.decode(answer_ids[0], skip_special_tokens=True)
```

Commentary: Production RAG needs chunking, deduplication, metadata filters, reranking, citation checking, and prompt-length control. The toy version is enough to show the core loop.

## 6. Implement Speculative Decoding

Speculative decoding uses a cheap draft model to propose tokens and a larger target model to verify them.

Simplified version:

```python
@torch.no_grad()
def speculative_decode(target_model, draft_model, input_ids, max_new_tokens, draft_steps):
    # input_ids: [B, T]
    tokens = input_ids

    while tokens.size(1) < input_ids.size(1) + max_new_tokens:
        # Draft proposes several tokens.
        draft = tokens
        proposed = []

        for _ in range(draft_steps):
            draft_logits = draft_model(draft)[:, -1, :]  # [B, Vocab]
            next_token = torch.argmax(draft_logits, dim=-1, keepdim=True)  # [B, 1]
            proposed.append(next_token)
            draft = torch.cat([draft, next_token], dim=-1)  # [B, T_current + 1]

        proposed_tokens = torch.cat(proposed, dim=-1)  # [B, draft_steps]
        candidate = torch.cat([tokens, proposed_tokens], dim=-1)  # [B, T + draft_steps]

        target_logits = target_model(candidate)  # [B, T + draft_steps, Vocab]

        accepted = []
        start = tokens.size(1)

        for i in range(draft_steps):
            verify_logits = target_logits[:, start + i - 1, :]  # [B, Vocab]
            target_token = torch.argmax(verify_logits, dim=-1, keepdim=True)  # [B, 1]

            if torch.equal(target_token, proposed_tokens[:, i:i+1]):
                accepted.append(proposed_tokens[:, i:i+1])  # [B, 1]
            else:
                accepted.append(target_token)  # [B, 1]
                break

        tokens = torch.cat([tokens] + accepted, dim=-1)  # [B, T + accepted_count]

    return tokens[:, :input_ids.size(1) + max_new_tokens]
```

Commentary: This simplified version uses greedy acceptance and does not preserve the exact sampling distribution. Production speculative decoding uses a careful accept/reject rule so outputs match the target model's distribution.

## 7. Compare RoPE vs ALiBi

A research coding task may ask you to implement both and compare validation loss or long-context retrieval.

RoPE:

```python
def rope_cache(seq_len, dim, device, base=10000):
    # seq_len: T
    # dim: D
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # inv_freq: [D/2]
    pos = torch.arange(seq_len, device=device).float()  # [T]
    angles = pos[:, None] * inv_freq[None, :]           # [T, D/2]
    return torch.cos(angles), torch.sin(angles)         # each [T, D/2]

def apply_rope(x, cos, sin):
    # x: [B, H, T, D]
    cos = cos[None, None, :, :]  # [1, 1, T, D/2]
    sin = sin[None, None, :, :]  # [1, 1, T, D/2]
    x1 = x[..., 0::2]            # [B, H, T, D/2]
    x2 = x[..., 1::2]            # [B, H, T, D/2]
    y1 = x1 * cos - x2 * sin     # [B, H, T, D/2]
    y2 = x1 * sin + x2 * cos     # [B, H, T, D/2]
    return torch.stack([y1, y2], dim=-1).flatten(-2)  # [B, H, T, D]
```

ALiBi:

```python
def alibi_bias(n_heads, seq_len, slopes, device):
    # slopes: [H]
    q_pos = torch.arange(seq_len, device=device)[:, None]  # [T, 1]
    k_pos = torch.arange(seq_len, device=device)[None, :]  # [1, T]
    distance = (q_pos - k_pos).clamp(min=0)                # [T, T]
    bias = -slopes[:, None, None] * distance[None, :, :]   # [H, T, T]
    return bias
```

Attention with either:

```python
def attention_with_position(q, k, v, mode, slopes=None):
    # q, k, v: [B, H, T, D]
    B, H, T, D = q.shape

    if mode == "rope":
        cos, sin = rope_cache(T, D, q.device)
        q = apply_rope(q, cos, sin)  # [B, H, T, D]
        k = apply_rope(k, cos, sin)  # [B, H, T, D]
        bias = 0
    elif mode == "alibi":
        bias = alibi_bias(H, T, slopes, q.device)[None, :, :, :]  # [1, H, T, T]
    else:
        bias = 0

    scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
    scores = scores / math.sqrt(D) + bias
    weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
    return weights @ v  # [B, H, T, D]
```

Comparison experiment:

```text
Train two tiny models:
  model A: RoPE
  model B: ALiBi

Keep fixed:
  parameter count
  data
  optimizer
  token budget
  random seeds if possible

Compare:
  validation loss
  length extrapolation
  retrieval accuracy
  training stability
```

## 8. Train a Tiny Transformer From Scratch

Tiny decoder-only model:

```python
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, hidden_dim, n_layers, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        # input_ids: [B, T]
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)  # [T]

        x = self.token_emb(input_ids)  # [B, T, C]
        x = x + self.pos_emb(pos)[None, :, :]  # [B, T, C]

        for block in self.blocks:
            x = block(x)  # [B, T, C]

        x = self.norm(x)        # [B, T, C]
        return self.lm_head(x)  # [B, T, Vocab]
```

Training loop:

```python
def train_tiny_transformer(model, loader, optimizer, num_steps):
    model.train()

    for step, batch in enumerate(loader):
        if step >= num_steps:
            break

        input_ids = batch["input_ids"].cuda()  # [B, T]

        logits = model(input_ids[:, :-1])  # [B, T-1, Vocab]
        targets = input_ids[:, 1:]         # [B, T-1]

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),  # [B*(T-1), Vocab]
            targets.reshape(-1),                  # [B*(T-1)]
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            print(step, loss.item())
```

Commentary: For a real from-scratch experiment, you also need a tokenizer, dataset packing, validation loop, checkpointing, and fixed random seeds.

## 9. Implement Sparse Attention

Sparse attention restricts which query-key pairs are allowed.

Sliding-window sparse mask:

```python
def sliding_window_mask(seq_len, window, device):
    # seq_len: T
    q_pos = torch.arange(seq_len, device=device)[:, None]  # [T, 1]
    k_pos = torch.arange(seq_len, device=device)[None, :]  # [1, T]

    causal = k_pos <= q_pos                  # [T, T]
    local = k_pos >= (q_pos - window)        # [T, T]
    return causal & local                    # [T, T]
```

Sparse attention with mask:

```python
def sparse_attention(q, k, v, window):
    # q, k, v: [B, H, T, D]
    B, H, T, D = q.shape

    scores = q @ k.transpose(-2, -1)  # [B, H, T, T]
    scores = scores / math.sqrt(D)

    mask = sliding_window_mask(T, window, q.device)  # [T, T]
    scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))  # [B, H, T, T]

    weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
    out = weights @ v                        # [B, H, T, D]
    return out
```

Block-sparse sketch:

```python
def block_sparse_attention(q_blocks, k_blocks, v_blocks, allowed):
    # q_blocks[i]: [B, H, block_T, D]
    outputs = []

    for i, q_block in enumerate(q_blocks):
        selected_k = torch.cat([k_blocks[j] for j in allowed[i]], dim=2)
        selected_v = torch.cat([v_blocks[j] for j in allowed[i]], dim=2)
        # selected_k, selected_v: [B, H, selected_T, D]

        scores = q_block @ selected_k.transpose(-2, -1)  # [B, H, block_T, selected_T]
        weights = torch.softmax(scores / math.sqrt(q_block.size(-1)), dim=-1)
        out = weights @ selected_v  # [B, H, block_T, D]
        outputs.append(out)

    return torch.cat(outputs, dim=2)  # [B, H, T, D]
```

Commentary: A mask-based sparse attention still materializes `[B, H, T, T]`; it is conceptually sparse but not computationally efficient. Real sparse attention needs block-sparse kernels.

## 10. Build a Toy MoE Model

A toy MoE replaces a dense MLP with routed expert MLPs.

Expert:

```python
class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False),
        )

    def forward(self, x):
        # x: [tokens_for_expert, C]
        return self.net(x)  # [tokens_for_expert, C]
```

MoE layer:

```python
class ToyMoE(nn.Module):
    def __init__(self, d_model, hidden_dim, n_experts, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(d_model, hidden_dim)
            for _ in range(n_experts)
        ])
        self.n_experts = n_experts
        self.top_k = top_k

    def forward(self, x):
        # x: [B, T, C]
        router_logits = self.router(x)  # [B, T, E]
        top_logits, top_idx = torch.topk(router_logits, self.top_k, dim=-1)
        # top_logits, top_idx: [B, T, K]

        top_weights = torch.softmax(top_logits, dim=-1)  # [B, T, K]
        out = torch.zeros_like(x)                        # [B, T, C]

        for slot in range(self.top_k):
            expert_ids = top_idx[..., slot]                 # [B, T]
            expert_weight = top_weights[..., slot][..., None] # [B, T, 1]

            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id  # [B, T]

                if mask.any():
                    x_e = x[mask]  # [tokens_for_expert, C]
                    y_e = expert(x_e)  # [tokens_for_expert, C]
                    out[mask] += expert_weight[mask] * y_e

        aux_loss = self.load_balance_loss(router_logits)
        return out, aux_loss  # [B, T, C], scalar

    def load_balance_loss(self, router_logits):
        # router_logits: [B, T, E]
        probs = torch.softmax(router_logits, dim=-1)  # [B, T, E]
        prob_per_expert = probs.mean(dim=(0, 1))      # [E]
        return self.n_experts * prob_per_expert.square().sum()
```

Use inside a transformer block:

```python
# x: [B, T, C]
moe_out, aux_loss = moe_layer(x)  # [B, T, C], scalar
x = x + moe_out                   # [B, T, C]
loss = lm_loss + 0.01 * aux_loss
```

Commentary: This implementation is intentionally simple. Production MoE must handle capacity factors, dropped tokens, expert parallelism, all-to-all communication, and efficient grouped matmuls.

## Final Research-Coding Checklist

You should now be able to implement and explain:

- Chinchilla compute estimation
- scaling-law fitting
- evaluation harnesses
- perplexity computation
- retrieval-augmented generation
- speculative decoding
- RoPE vs ALiBi experiments
- tiny transformer training
- sparse attention
- toy MoE routing and auxiliary loss

For research coding interviews, always explain what is simplified, what metric you would report, and what could make the result misleading.
