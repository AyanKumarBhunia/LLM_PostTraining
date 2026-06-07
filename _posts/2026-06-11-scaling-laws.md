---
title: "Section E: Scaling Laws"
date: 2026-06-11 12:00:00 +0000
series_order: 5
categories: [interview-prep]
tags: [llm, scaling-laws, chinchilla, kaplan, compute-optimal-training]
pin: false
math: false
permalink: /posts/scaling-laws/
---

## Goal

This is the fifth section of the frontier-model Training Team interview preparation series.

In this tutorial, we cover scaling laws:

1. What scaling laws are
2. Kaplan scaling laws
3. Chinchilla scaling laws
4. Why many early LLMs were undertrained
5. Whether to scale parameters or data under fixed compute
6. Compute-optimal training
7. Why scaling laws matter
8. What causes scaling-law deviations
9. How to estimate optimal token count
10. Whether scaling laws continue forever

Throughout the code:

- `N` = number of model parameters
- `D` = number of training tokens
- `C` = training compute budget in FLOPs
- `L` = loss
- `B` = batch size
- `T` = sequence length
- `Vocab` = vocabulary size

## 61. What Are Scaling Laws?

Scaling laws are empirical relationships between model performance and quantities like:

- model size
- dataset size
- training compute
- loss

For LLMs, the central observation is:

> As you scale parameters, data, and compute, validation loss often improves predictably over a wide range.

A simple conceptual form:

```python
# This is a toy power-law form, not a universal exact formula.
def predicted_loss(scale, irreducible_loss, coefficient, exponent):
    # scale: scalar, e.g. parameter count or token count
    # output: scalar predicted loss
    return irreducible_loss + coefficient * (scale ** (-exponent))
```

The important idea is that loss often decreases smoothly rather than randomly.

Example:

```text
10M params  -> higher loss
100M params -> lower loss
1B params   -> lower loss
10B params  -> lower loss
```

Scaling laws matter because they let teams predict what may happen before spending millions of GPU hours.

Commentary: Scaling laws are empirical. They describe observed regimes, not laws of physics.

## 62. Explain Kaplan Scaling Laws

Kaplan et al. studied how language-model loss scales with:

- model size
- dataset size
- compute

The broad result:

> Loss follows approximate power laws as model size, data size, and compute increase.

Toy conceptual model:

```python
def kaplan_style_loss(N, D, A=1.0, B=1.0, alpha=0.08, beta=0.10, L_inf=1.5):
    # N: parameter count, scalar
    # D: training tokens, scalar
    # Returns toy predicted validation loss.
    model_term = A * (N ** (-alpha))
    data_term = B * (D ** (-beta))
    return L_inf + model_term + data_term
```

This says loss can improve by:

- increasing `N`
- increasing `D`
- increasing both

Kaplan-style takeaway:

- Bigger models are very powerful.
- Performance improves predictably with scale.
- Large models can be more sample-efficient.

Important interview nuance:

> Kaplan-era recommendations tended to favor training larger models on fewer tokens than later Chinchilla-style compute-optimal results.

Commentary: Do not quote exact exponents unless you are sure. In interviews, the safer answer is to explain the qualitative result and the compute-allocation implication.

## 63. Explain Chinchilla Scaling Laws

Chinchilla scaling laws revisited compute-optimal training.

The headline result:

> Many large models were too big for the number of tokens they were trained on. For fixed compute, it is often better to train a smaller model on more tokens.

Back-of-the-envelope dense transformer compute:

```python
def training_flops(params, tokens):
    # params: N, scalar
    # tokens: D, scalar
    # Common rough estimate for dense decoder-only transformers.
    return 6 * params * tokens
```

If compute is fixed:

```python
compute_budget = 6 * N * D
```

Increasing parameters means you must reduce tokens unless compute also increases.

Chinchilla-style intuition:

```text
Do not only ask:
  How big can the model be?

Ask:
  What model size and token count minimize loss under my compute budget?
```

Chinchilla found that compute-optimal models should be trained on far more tokens than many earlier large models used.

Interview answer:

> Chinchilla showed that, under a fixed compute budget, many earlier LLMs were undertrained on data. A smaller model trained on more tokens could outperform a larger model trained on too few tokens.

## 64. Why Were Many Early LLMs Undertrained?

Many early LLMs prioritized parameter count.

The implicit strategy was:

```text
Make N very large.
Train on available D.
```

But compute is roughly:

```python
C = 6 * N * D
```

If `C` is fixed and `N` is very large, then `D` must be smaller.

That leads to undertraining:

```text
model has enough parameters to learn more,
but it does not see enough tokens.
```

Symptoms of undertraining:

- validation loss could keep improving with more tokens
- model capacity is high but not fully used
- smaller models trained longer may match or beat it

Example:

```python
def tokens_possible(compute_budget, params):
    # compute_budget: C
    # params: N
    return compute_budget / (6 * params)

C = 1e24
tokens_for_100b = tokens_possible(C, 100e9)
tokens_for_10b = tokens_possible(C, 10e9)

# tokens_for_10b is 10x larger than tokens_for_100b under the same compute.
```

Interview answer:

> Many early LLMs were undertrained because teams scaled parameter count faster than token count. Chinchilla-style results showed that those models could have used more data, or that smaller models trained on more tokens could be more compute-optimal.

## 65. Given Fixed Compute, Should You Scale Parameters or Data?

The correct answer is:

> It depends where you are relative to the compute-optimal frontier.

With fixed compute:

```python
C = 6 * N * D
```

If you increase `N`, you must decrease `D`:

```python
def tokens_for_params(C, N):
    # C: compute budget
    # N: parameter count
    return C / (6 * N)
```

A simple sweep:

```python
candidate_params = torch.tensor([1e9, 3e9, 7e9, 13e9, 30e9])  # [num_candidates]
compute_budget = 1e23

candidate_tokens = compute_budget / (6 * candidate_params)  # [num_candidates]
```

You would estimate loss for each `(N, D)` pair:

```python
def toy_loss(N, D):
    # N, D: scalar or tensor with same shape
    return 1.5 + 2.0 * N.pow(-0.08) + 1.5 * D.pow(-0.10)

losses = toy_loss(candidate_params, candidate_tokens)  # [num_candidates]
best_idx = losses.argmin()
```

In a real setting, you would use fitted scaling laws, smaller pilot runs, and system constraints.

Interview answer:

> Given fixed compute, I would not blindly scale parameters. I would choose the model size and token count that are estimated to minimize loss under the compute budget. Chinchilla suggests many regimes benefit from more data relative to parameters than earlier practice used.

## 66. What Is Compute-Optimal Training?

Compute-optimal training means choosing model size and data size to get the best performance for a fixed compute budget.

Given:

```text
C = compute budget
```

Choose:

```text
N = number of parameters
D = number of tokens
```

to minimize expected loss.

Toy code:

```python
def estimate_compute(N, D):
    # N: parameter count
    # D: token count
    return 6 * N * D

def estimate_loss(N, D):
    # Toy power-law loss, not a real fitted law.
    return 1.5 + 2.0 * (N ** -0.08) + 1.5 * (D ** -0.10)

def search_compute_optimal(compute_budget, candidate_params):
    # candidate_params: [num_candidates]
    candidate_tokens = compute_budget / (6 * candidate_params)  # [num_candidates]
    losses = estimate_loss(candidate_params, candidate_tokens)  # [num_candidates]
    best = losses.argmin()
    return candidate_params[best], candidate_tokens[best], losses[best]
```

Compute-optimal does not always mean globally optimal for product use.

Other constraints matter:

- inference cost
- latency
- memory
- available data quality
- training stability
- deployment hardware
- desired capability profile

Commentary: A compute-optimal training run for validation loss may not be optimal for serving cost. Training teams must consider both.

## 67. Why Do Scaling Laws Matter?

Scaling laws matter because large training runs are expensive.

They help answer:

- How large should the model be?
- How many tokens should we train on?
- How much compute do we need?
- Is this architecture improvement real?
- Are we data-limited or model-limited?
- What loss should we expect from a bigger run?

Example use:

```python
# Results from pilot runs:
params = torch.tensor([100e6, 300e6, 1e9, 3e9])  # [runs]
tokens = torch.tensor([10e9, 30e9, 100e9, 300e9])  # [runs]
loss = torch.tensor([3.1, 2.8, 2.45, 2.25])  # [runs]

# Fit a simple scaling model, then extrapolate to larger runs.
```

In practice, teams use scaling laws to:

- plan compute budgets
- avoid wasting training runs
- compare architectures at small scale
- predict whether more data or parameters will help
- estimate expected returns from scaling

Interview answer:

> Scaling laws turn model development from pure trial-and-error into prediction. They help allocate compute, data, and model size before launching expensive frontier-scale runs.

## 68. What Might Cause Scaling-Law Deviations?

Scaling laws can break or shift when assumptions change.

Common causes:

- Data quality changes
- Data duplication or contamination
- Tokenizer changes
- Architecture changes
- Optimizer or schedule changes
- Batch size changes
- Context length changes
- Training instability
- Evaluation mismatch
- Model enters a new capability regime
- Too much extrapolation beyond measured scales

Example:

```python
# Suppose pilot runs use clean web data.
pilot_data_quality = "clean"

# Large run accidentally includes duplicated low-quality data.
large_run_data_quality = "duplicated_noisy"

# The observed loss may be worse than predicted.
```

Architecture deviations:

```text
Dense transformer scaling law
  may not directly predict
MoE transformer scaling
```

Context-length deviations:

```text
2k context scaling
  may not directly predict
128k context scaling
```

Interview answer:

> Deviations can come from changing data, architecture, optimizer, tokenizer, context length, evaluation, or stability. Scaling laws are only reliable when the new run stays close to the regime used to fit them.

## 69. How Would You Estimate Optimal Token Count?

A simple first estimate uses the dense transformer compute rule:

```python
C = 6 * N * D
```

Given compute budget and model size:

```python
def estimate_tokens_from_compute(compute_budget, params):
    # compute_budget: C
    # params: N
    return compute_budget / (6 * params)
```

But this only gives the token count that fits compute for a chosen `N`. It does not prove that `N` is optimal.

A better approach:

1. Fit or use a scaling-law model.
2. Sweep candidate model sizes.
3. Compute the token count each model can afford.
4. Pick the pair with lowest predicted loss.

Code:

```python
def estimate_training_flops(params, tokens):
    # params: N
    # tokens: D
    return 6 * params * tokens

def predicted_loss(params, tokens):
    # Toy example only.
    # params and tokens can be tensors with shape [num_candidates].
    return 1.5 + 2.0 * params.pow(-0.08) + 1.5 * tokens.pow(-0.10)

compute_budget = 1e24
candidate_params = torch.tensor([3e9, 7e9, 13e9, 30e9, 70e9])  # [5]
candidate_tokens = compute_budget / (6 * candidate_params)     # [5]

losses = predicted_loss(candidate_params, candidate_tokens)     # [5]
best = torch.argmin(losses)

best_params = candidate_params[best]  # scalar
best_tokens = candidate_tokens[best]  # scalar
```

Practical additions:

- Check whether enough high-quality tokens exist.
- Account for repeated epochs over data.
- Account for sequence length and packing efficiency.
- Account for hardware utilization.
- Account for inference constraints.

Chinchilla-style rule of thumb is often summarized as training on many more tokens per parameter than earlier GPT-3-style practice.

Commentary: Exact "optimal token count" depends on the fitted scaling law, data quality, architecture, and compute accounting. Use the formula as a planning estimate, not a guarantee.

## 70. Do Scaling Laws Continue Forever?

No one knows.

The careful answer:

> Scaling laws have held across large observed ranges, but they are empirical trends. They may change when data, architecture, compute, or task distributions change.

Possible reasons scaling may not continue forever:

- finite high-quality data
- data contamination and duplication
- irreducible loss from ambiguous prediction
- new bottlenecks from reasoning, tool use, or agency
- optimization limits
- architecture limits
- economic and hardware constraints
- evaluation saturation

Toy view:

```python
def loss_with_floor(scale):
    # scale: scalar
    irreducible_loss = 1.2
    return irreducible_loss + 10.0 * scale ** -0.1
```

As scale grows, loss approaches a floor:

```text
loss -> irreducible_loss
```

But new capabilities may also appear with scale, and new bottlenecks may appear too.

Interview answer:

> I would not assume scaling laws continue forever unchanged. They are useful empirical guides, but they can bend or break because of data limits, architecture changes, optimization issues, and evaluation saturation.

## Final Interview Checklist

You should now be able to answer all Section E questions:

- Scaling laws are empirical relationships between scale and loss.
- Kaplan showed predictable power-law improvement with parameters, data, and compute.
- Chinchilla showed many models should be trained on more tokens for fixed compute.
- Early LLMs were often undertrained because parameter count was scaled faster than data.
- With fixed compute, choose parameters and data jointly.
- Compute-optimal training minimizes loss under a compute budget.
- Scaling laws matter because they guide expensive training decisions.
- Deviations can come from data, architecture, optimizer, tokenizer, context, or instability.
- Optimal token count can be estimated by combining compute accounting with fitted loss models.
- Scaling laws are empirical and may not continue forever unchanged.
