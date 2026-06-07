---
title: "Section H: Research and Experiment Design"
date: 2026-06-13 12:00:00 +0000
series_order: 7
categories: [interview-prep]
tags: [llm, research, evaluation, ablation, benchmark, regression, experiment-design]
pin: false
math: false
permalink: /posts/research-experiment-design/
---

## Goal

This is the seventh tutorial of the frontier-model Training Team interview preparation series.

In this tutorial, we cover research and experiment design:

1. How to evaluate a new architecture
2. How to know an improvement is real
3. What ablations to run
4. How to detect benchmark overfitting
5. How to debug a regression
6. Why perplexity can improve while coding benchmarks drop
7. How to evaluate reasoning ability
8. How to design an experiment for long-context reasoning
9. How to talk about a failed research project
10. What LLM research direction is exciting and why

Throughout the code:

- `B` = batch size
- `T` = sequence length
- `C` = model width
- `Vocab` = vocabulary size
- `N` = number of evaluation examples
- `K` = number of benchmark tasks or metrics
- `R` = number of random seeds or repeated runs

## 91. How Do You Evaluate a New Architecture?

Evaluating a new architecture is not just checking whether the final benchmark number is higher. A good evaluation asks:

- Is the new architecture better at fixed compute?
- Is it better at fixed parameter count?
- Is it better at fixed inference cost?
- Is it stable across scales?
- Does it improve the target capability, or only one proxy metric?

For LLMs, architecture changes can affect many things at once:

- training loss
- validation loss
- downstream benchmarks
- inference latency
- memory use
- context-length behavior
- stability
- ease of implementation

A clean architecture experiment starts with a strong baseline.

```python
class BaselineBlock(nn.Module):
    def __init__(self, d_model, attn, mlp):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = attn
        self.mlp = mlp

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.norm1(x))  # [B, T, C]
        x = x + self.mlp(self.norm2(x))   # [B, T, C]
        return x                          # [B, T, C]
```

Then change one thing.

```python
class CandidateBlock(nn.Module):
    def __init__(self, d_model, new_attn, mlp):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = new_attn
        self.mlp = mlp

    def forward(self, x):
        # x: [B, T, C]
        x = x + self.attn(self.norm1(x))  # [B, T, C]
        x = x + self.mlp(self.norm2(x))   # [B, T, C]
        return x                          # [B, T, C]
```

The comparison should control as many variables as possible:

```python
def train_architecture(model, dataloader, optimizer, max_tokens):
    # max_tokens: scalar budget, e.g. 50B tokens
    tokens_seen = 0

    while tokens_seen < max_tokens:
        batch = next(dataloader)
        input_ids = batch["input_ids"]      # [B, T]
        labels = batch["labels"]            # [B, T]

        logits = model(input_ids)           # [B, T, Vocab]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # [B*T, Vocab]
            labels.view(-1),                   # [B*T]
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tokens_seen += input_ids.numel()

    return model
```

Key evaluation axes:

```python
metrics = {
    "validation_loss": 1.92,
    "mmlu": 0.712,
    "gsm8k": 0.648,
    "humaneval": 0.402,
    "tokens_per_second": 185000,
    "peak_memory_gb": 72.4,
}
```

A useful architecture change should have a clear story. For example:

> This attention variant gives the same validation loss with 8% less training compute and 12% faster inference, without hurting reasoning or coding benchmarks.

Interview answer:

> I would evaluate a new architecture against a strong baseline under matched compute, data, optimizer, and training schedule. I would compare loss curves, downstream benchmarks, inference cost, memory use, scaling behavior, and stability across seeds and model sizes. The goal is not only to show a better number, but to understand the tradeoff the architecture creates.

Commentary: A new architecture can look good at small scale and fail at larger scale. I would be careful about claiming success from one model size or one benchmark.

## 92. How Do You Know an Improvement Is Real?

An improvement is real when it survives noise, repeated runs, and alternative evaluations.

LLM experiments are noisy because of:

- random initialization
- data ordering
- nondeterministic kernels
- benchmark sampling variance
- prompt sensitivity
- evaluation harness bugs
- checkpoint selection

Suppose we compare a baseline and a candidate across repeated runs:

```python
# baseline_scores: [R]
# candidate_scores: [R]
baseline_scores = torch.tensor([0.421, 0.426, 0.419, 0.424])
candidate_scores = torch.tensor([0.437, 0.431, 0.440, 0.436])

delta = candidate_scores.mean() - baseline_scores.mean()  # scalar
baseline_std = baseline_scores.std(unbiased=True)         # scalar
candidate_std = candidate_scores.std(unbiased=True)       # scalar
```

The improvement is more convincing if:

- the candidate wins across multiple seeds
- confidence intervals do not overlap much
- the result appears on independent evals
- the training curves separate early and stay separated
- the change matches the expected mechanism
- the improvement is large enough to matter in practice

A simple paired evaluation is often better than comparing unrelated runs.

```python
def paired_accuracy_delta(model_a, model_b, eval_batch):
    input_ids = eval_batch["input_ids"]  # [N, T]
    labels = eval_batch["labels"]        # [N]

    with torch.no_grad():
        logits_a = model_a(input_ids)    # [N, num_choices]
        logits_b = model_b(input_ids)    # [N, num_choices]

    pred_a = logits_a.argmax(dim=-1)     # [N]
    pred_b = logits_b.argmax(dim=-1)     # [N]

    correct_a = pred_a.eq(labels).float()  # [N]
    correct_b = pred_b.eq(labels).float()  # [N]

    delta = (correct_b - correct_a).mean() # scalar
    return delta
```

Why paired comparison helps:

- both models see the same examples
- example difficulty is controlled
- the variance of the difference is usually lower than the variance of raw scores

For generative benchmarks, also control decoding:

```python
generation_config = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
}
```

If temperature is nonzero, run multiple samples per prompt.

```python
def pass_at_k(successes):
    # successes: [N, K], bool for whether sample k solved example n
    solved = successes.any(dim=-1)       # [N]
    return solved.float().mean()         # scalar
```

Interview answer:

> I would treat an improvement as real only if it is larger than expected noise, repeats across seeds or shards, holds under matched evaluation settings, and improves more than one narrow metric. I would use paired evaluations, confidence intervals, and independent benchmarks before making a strong claim.

Commentary: I am not sure there is one universal threshold for "real." The threshold depends on cost, benchmark variance, deployment importance, and how surprising the claim is.

## 93. What Ablations Would You Run?

Ablations identify which part of a change actually matters.

If a new method includes three changes:

1. new architecture
2. new data mixture
3. new learning-rate schedule

Then a single "baseline vs new model" comparison does not tell us what helped.

A basic ablation table:

```text
baseline architecture + old data + old schedule
new architecture      + old data + old schedule
baseline architecture + new data + old schedule
baseline architecture + old data + new schedule
new architecture      + new data + old schedule
new architecture      + old data + new schedule
new architecture      + new data + new schedule
```

In code, think of the experiment as a config sweep:

```python
configs = [
    {"arch": "baseline", "data": "old", "lr_schedule": "old"},
    {"arch": "candidate", "data": "old", "lr_schedule": "old"},
    {"arch": "baseline", "data": "new", "lr_schedule": "old"},
    {"arch": "baseline", "data": "old", "lr_schedule": "new"},
    {"arch": "candidate", "data": "new", "lr_schedule": "new"},
]
```

Common ablations for LLM architecture work:

- remove the new component entirely
- vary component size
- vary where the component is inserted
- vary training duration
- test multiple model scales
- test multiple data mixtures
- test multiple context lengths
- test different inference settings

Example: ablate a new attention mechanism.

```python
def build_model(attn_type, d_model, n_layers):
    blocks = []

    for _ in range(n_layers):
        if attn_type == "standard":
            attn = StandardAttention(d_model)
        elif attn_type == "candidate":
            attn = CandidateAttention(d_model)
        elif attn_type == "candidate_no_gate":
            attn = CandidateAttention(d_model, use_gate=False)
        else:
            raise ValueError(attn_type)

        blocks.append(BaselineBlock(d_model, attn, SwiGLU(d_model)))

    return Transformer(blocks)
```

Ablate scale:

```python
model_sizes = [
    {"layers": 12, "d_model": 768},    # small
    {"layers": 24, "d_model": 1024},   # medium
    {"layers": 32, "d_model": 4096},   # large
]
```

Ablate context:

```python
context_lengths = [2048, 8192, 32768, 131072]

for T in context_lengths:
    input_ids = torch.randint(0, vocab_size, (B, T))  # [B, T]
    logits = model(input_ids)                         # [B, T, Vocab]
```

Important ablation question:

> Is the new method still useful when the baseline is tuned equally well?

Many weak papers compare a carefully tuned new method against an undertuned baseline. In strong research, the baseline gets the same level of care.

Interview answer:

> I would ablate each proposed change separately, then test important interactions. I would vary model scale, training tokens, data mixture, context length, and inference settings. I would also tune the baseline fairly, because otherwise the ablation only proves that the new method was tuned better.

Commentary: Full ablations can be expensive. In real teams, I would start with cheap small-scale ablations, then promote only the most informative experiments to larger scale.

## 94. How Do You Detect Benchmark Overfitting?

Benchmark overfitting happens when a model or research process becomes too optimized for a benchmark and stops reflecting general ability.

This can happen through:

- benchmark examples appearing in training data
- repeated manual tuning against the same benchmark
- prompt engineering specifically for one benchmark
- selecting checkpoints based on benchmark score
- optimizing decoding settings for one leaderboard
- learning benchmark-specific artifacts

First, check contamination.

```python
def ngram_overlap(train_tokens, eval_tokens, n):
    # train_tokens: [num_train_tokens]
    # eval_tokens: [num_eval_tokens]
    train_ngrams = set()

    for i in range(len(train_tokens) - n + 1):
        train_ngrams.add(tuple(train_tokens[i:i+n].tolist()))

    matches = 0
    total = 0
    for i in range(len(eval_tokens) - n + 1):
        total += 1
        if tuple(eval_tokens[i:i+n].tolist()) in train_ngrams:
            matches += 1

    return matches / max(total, 1)
```

For large corpora, exact set matching is too expensive, so teams use approximate methods such as MinHash, suffix arrays, or embedding-based retrieval.

Second, compare public and private evals.

```python
# scores: [K], one score per benchmark
public_scores = torch.tensor([0.82, 0.76, 0.71, 0.68])   # [K]
private_scores = torch.tensor([0.70, 0.73, 0.69, 0.67])  # [K]

gap = public_scores - private_scores                     # [K]
mean_gap = gap.mean()                                    # scalar
```

Large public-private gaps can suggest overfitting, especially when the public benchmark was repeatedly used during development.

Third, use benchmark variants.

```python
def perturb_prompt(prompt):
    return [
        prompt,
        "Solve carefully:\n" + prompt,
        prompt.replace("Answer:", "Final answer:"),
        prompt + "\nExplain your reasoning before answering.",
    ]
```

If a model only works under one exact prompt format, the benchmark score may not represent robust ability.

Fourth, create fresh held-out evaluations.

Good held-out evals:

- are not used during model selection
- are built after the training cutoff
- include adversarial or paraphrased examples
- are checked for leakage
- measure the intended capability directly

Interview answer:

> I would detect benchmark overfitting by checking training-data contamination, comparing public and private benchmark performance, using fresh held-out evals, testing prompt and format robustness, and looking for unnatural score spikes on heavily optimized benchmarks. A model that improves only on the leaderboard but not on neighboring tasks is suspicious.

Commentary: High benchmark performance is not automatically overfitting. The warning sign is a pattern: narrow gains, leakage risk, prompt brittleness, and weak transfer to related tasks.

## 95. How Do You Debug a Regression?

A regression means the new system is worse than the old system on some metric or behavior.

The first step is to localize the regression:

- Did training loss regress?
- Did validation loss regress?
- Did only one benchmark regress?
- Did inference behavior change?
- Did the evaluation harness change?
- Did the data change?
- Did the checkpoint or tokenizer change?

Start by confirming the regression is real.

```python
def evaluate_loss(model, batch):
    input_ids = batch["input_ids"]  # [B, T]
    labels = batch["labels"]        # [B, T]

    with torch.no_grad():
        logits = model(input_ids)   # [B, T, Vocab]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),  # [B*(T-1), Vocab]
            labels[:, 1:].reshape(-1),                       # [B*(T-1)]
        )

    return loss                     # scalar
```

Then compare old and new models on the same examples.

```python
def per_example_nll(model, input_ids, labels):
    # input_ids: [B, T]
    # labels: [B, T]
    with torch.no_grad():
        logits = model(input_ids)  # [B, T, Vocab]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, Vocab]

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),  # [B, T, 1]
    ).squeeze(-1)                    # [B, T]

    nll = -token_log_probs.mean(dim=-1)  # [B]
    return nll

old_nll = per_example_nll(old_model, input_ids, labels)  # [B]
new_nll = per_example_nll(new_model, input_ids, labels)  # [B]
delta = new_nll - old_nll                                # [B]
worst_examples = delta.topk(k=8).indices                 # [8]
```

Looking at the worst-regressed examples often reveals the failure mode:

- code formatting got worse
- long prompts degrade
- math word problems degrade
- non-English examples degrade
- model refuses too often
- tokenizer behavior changed
- generation stops too early

Debugging checklist:

```text
1. Re-run evaluation with fixed seeds.
2. Check whether the evaluation data changed.
3. Check whether prompts/templates changed.
4. Compare exact model configs.
5. Compare tokenizer and chat template.
6. Compare training data mixture.
7. Compare loss curves by domain.
8. Inspect per-example regressions.
9. Run ablations to isolate the responsible change.
10. Reproduce at small scale if possible.
```

For training regressions, inspect curves.

```python
# train_loss: [num_steps]
# val_loss: [num_checkpoints]
train_loss_delta = new_train_loss - old_train_loss  # [num_steps]
val_loss_delta = new_val_loss - old_val_loss        # [num_checkpoints]
```

If training loss is worse, suspect optimization or implementation. If training loss is better but benchmark is worse, suspect data, capability tradeoffs, or evaluation mismatch.

Interview answer:

> I would first verify the regression under identical evaluation settings, then localize it by metric, domain, checkpoint, and example. I would compare configs, data, tokenizer, prompts, decoding settings, and per-example losses. Once localized, I would run small ablations to identify the responsible change.

Commentary: Many regressions are not model-quality regressions. They are evaluation-template changes, tokenizer mismatches, stop-sequence bugs, or checkpoint-selection mistakes.

## 96. Perplexity Improved but Coding Benchmarks Dropped. Why?

This is common because perplexity and coding benchmark performance measure different things.

Perplexity measures average next-token prediction quality.

```python
def perplexity_from_loss(loss):
    # loss: scalar cross-entropy in nats
    return torch.exp(loss)  # scalar
```

Coding benchmarks often measure functional correctness.

```python
def coding_eval(model, prompts, unit_tests):
    # prompts: list[str], length N
    # unit_tests: list[Callable], length N
    successes = []

    for prompt, tests in zip(prompts, unit_tests):
        code = model.generate(prompt, temperature=0.2)  # string
        passed = tests(code)                            # bool
        successes.append(passed)

    return torch.tensor(successes).float().mean()       # scalar
```

A model can assign higher probability to common code tokens while becoming worse at solving programming tasks.

Possible reasons:

- The data mixture shifted toward natural language and away from code.
- Code perplexity improved on easy code but not hard algorithmic code.
- The model got better at local syntax but worse at long-range program planning.
- The tokenizer or formatting changed in a way that hurts code generation.
- Alignment or instruction tuning made the model more verbose or more refusal-prone.
- Decoding settings changed.
- The benchmark requires execution correctness, not likely-looking code.
- The model overfits common boilerplate and underperforms on rare algorithms.

Example: token-level loss can improve while exact solution quality drops.

```python
# logits: [B, T, Vocab]
# labels: [B, T]
token_loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),  # [B*T, Vocab]
    labels.reshape(-1),                   # [B*T]
    reduction="none",
).reshape(B, T)                           # [B, T]

mean_loss = token_loss.mean()             # scalar
```

The mean loss may be dominated by easy tokens:

```python
# easy_mask: [B, T], True for syntax, whitespace, common identifiers
# hard_mask: [B, T], True for rare logic-heavy tokens
easy_loss = token_loss[easy_mask].mean()  # scalar
hard_loss = token_loss[hard_mask].mean()  # scalar
```

If easy loss improves but hard loss worsens, total perplexity can still look better.

Coding benchmarks also depend on sequence-level correctness:

```python
# token_correct: [B, T]
token_correct = logits.argmax(dim=-1).eq(labels)  # [B, T]

# A program can fail if one important token is wrong.
sequence_correct = token_correct.all(dim=-1)      # [B]
```

For code, one wrong operator, indentation level, variable name, or boundary condition can make the whole program fail.

How to investigate:

- Evaluate perplexity separately on code, math, natural language, and mixed data.
- Split code loss by language and difficulty.
- Run execution-based benchmarks.
- Inspect failed generations.
- Check formatting, stop tokens, and chat template.
- Compare pass@1 and pass@k.
- Test with deterministic and sampled decoding.

Interview answer:

> Perplexity can improve while coding drops because next-token likelihood is not the same as functional correctness. The model may improve on common or easy tokens while losing algorithmic planning, long-range consistency, formatting, or execution correctness. I would debug by evaluating code-specific loss, execution benchmarks, pass@k, decoding settings, and per-example failures.

Commentary: If perplexity improved on the general validation set, I would not assume code ability improved. I would need domain-specific and execution-based evidence.

## 97. How Would You Evaluate Reasoning Ability?

Reasoning ability is hard to evaluate because many benchmarks mix reasoning with:

- memorization
- language understanding
- tool familiarity
- arithmetic
- benchmark-specific formatting
- instruction following

A good reasoning evaluation should test whether the model can combine facts, track constraints, and solve problems it has not memorized.

Useful reasoning eval types:

- math word problems
- theorem-style problems
- logic puzzles
- causal reasoning
- multi-hop question answering
- planning tasks
- program synthesis
- counterfactual reasoning
- hidden-state or scratchpad consistency checks

A simple multiple-choice reasoning eval:

```python
def score_choices(model, input_ids, choice_token_ids):
    # input_ids: [B, T]
    # choice_token_ids: [B, num_choices]
    with torch.no_grad():
        logits = model(input_ids)              # [B, T, Vocab]

    final_logits = logits[:, -1, :]            # [B, Vocab]
    choice_logits = final_logits.gather(
        dim=-1,
        index=choice_token_ids,                # [B, num_choices]
    )                                          # [B, num_choices]

    return choice_logits                       # [B, num_choices]
```

For generative reasoning, exact answer extraction matters.

```python
def evaluate_numeric_answers(model, prompts, answers):
    # prompts: list[str], length N
    # answers: list[float], length N
    correct = []

    for prompt, gold in zip(prompts, answers):
        response = model.generate(prompt, temperature=0.0)  # string
        pred = extract_final_number(response)               # scalar or None
        correct.append(pred == gold)

    return torch.tensor(correct).float().mean()             # scalar
```

But final-answer accuracy is not enough. We also want robustness.

Reasoning robustness checks:

- paraphrase the problem
- change variable names
- change irrelevant surface details
- increase number of reasoning steps
- use generated-but-verified new problems
- test out-of-distribution difficulty
- require concise answers and full explanations separately

Example perturbation:

```python
def make_counterfactual_batch(problem_batch):
    # problem_batch["numbers"]: [B, M]
    # problem_batch["templates"]: list[str], length B
    numbers = problem_batch["numbers"]        # [B, M]
    new_numbers = numbers + torch.randint_like(numbers, low=1, high=5)  # [B, M]

    prompts = render_templates(problem_batch["templates"], new_numbers) # list[str], length B
    answers = solve_symbolically(new_numbers)                           # [B]

    return prompts, answers
```

A stronger evaluation uses verifiers:

```python
def verifier_score(solution, checker):
    # solution: generated string
    # checker: programmatic verifier
    return checker(solution)  # bool
```

This is useful for math, code, formal logic, and some planning tasks because it reduces dependence on string matching.

Interview answer:

> I would evaluate reasoning with diverse, held-out, contamination-checked tasks that require multi-step inference rather than memorized answers. I would use exact-answer and verifier-based metrics, test robustness to paraphrases and counterfactual changes, scale difficulty, and inspect failure modes. I would not rely on one benchmark as a complete measure of reasoning.

Commentary: I am not sure reasoning can be captured by a single score. It is better treated as a family of capabilities tested under controlled conditions.

## 98. Design an Experiment for Long-Context Reasoning

Long-context reasoning is not the same as long-context retrieval.

Retrieval asks:

> Can the model find the relevant fact in a long document?

Reasoning asks:

> Can the model find the relevant facts, combine them correctly, ignore distractors, and produce the right conclusion?

A good long-context reasoning experiment should vary:

- context length
- number of relevant facts
- distance between relevant facts
- number of distractors
- reasoning depth
- answer format
- position of evidence

Example synthetic task:

```text
In a long document, place facts like:

Alice is Bob's manager.
Bob is Clara's manager.
Clara approves Project Z.

Question:
Who is the top-level manager connected to the person who approves Project Z?
```

The model must retrieve multiple facts and compose them.

Dataset generator:

```python
def generate_long_context_example(T, num_facts, num_distractors, tokenizer):
    # T: target sequence length in tokens
    # num_facts: scalar
    # num_distractors: scalar

    relevant_facts, answer = sample_reasoning_chain(num_facts)  # list[str], string
    distractors = sample_distractors(num_distractors)           # list[str]

    document_parts = interleave_randomly(relevant_facts, distractors)  # list[str]
    document = "\n".join(document_parts)                              # string

    prompt = document + "\n\nQuestion: " + make_question(answer)       # string
    input_ids = tokenizer(prompt).input_ids                            # [T_actual]

    input_ids = pad_or_trim_to_length(input_ids, T)                    # [T]
    return input_ids, answer
```

Batch construction:

```python
input_ids = torch.stack([
    torch.tensor(generate_long_context_example(T=32768, num_facts=4, num_distractors=200, tokenizer=tokenizer)[0])
    for _ in range(B)
])  # [B, T]
```

Evaluation:

```python
def evaluate_long_context_reasoning(model, batch):
    input_ids = batch["input_ids"]      # [B, T]
    answers = batch["answers"]          # list[str], length B

    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        temperature=0.0,
    )                                   # [B, T + new_T]

    responses = tokenizer.batch_decode(outputs[:, input_ids.size(1):])  # list[str]
    correct = [extract_answer(r) == a for r, a in zip(responses, answers)]

    return torch.tensor(correct).float().mean()  # scalar
```

Important experimental controls:

- Keep the same reasoning chain while moving evidence positions.
- Test short-context versions to separate reasoning failure from context failure.
- Test retrieval-only versions to separate retrieval from composition.
- Add adversarial distractors with similar names or facts.
- Measure performance as a function of token distance.
- Check whether the model uses the full context or only recency.

Position sweep:

```python
positions = torch.linspace(0.05, 0.95, steps=10)  # [10]

for pos in positions:
    batch = make_position_controlled_batch(
        evidence_position=float(pos),
        context_length=65536,
        batch_size=B,
    )
    score = evaluate_long_context_reasoning(model, batch)  # scalar
```

A useful result plot would show accuracy over:

```text
context length x reasoning depth x evidence position x distractor count
```

Expected failure modes:

- lost-in-the-middle behavior
- confusing similar entities
- retrieving one fact but not composing all facts
- answering from priors instead of context
- truncation or positional extrapolation failure
- degraded instruction following at long context

Interview answer:

> I would design synthetic and natural long-context tasks where the answer requires retrieving multiple separated facts and composing them. I would vary context length, evidence position, distractor count, and reasoning depth. I would include retrieval-only and short-context controls so I can distinguish context access from reasoning ability.

Commentary: Synthetic tasks are useful because they give control, but they may not represent real documents. I would pair them with natural long-document QA, legal, codebase, or scientific-paper tasks.

## 99. Tell Me About a Failed Research Project

This interview question is not only about the project. It is about how you reason when an idea fails.

A strong answer should include:

- the hypothesis
- the experiment
- what happened
- how you debugged it
- what you learned
- what you would do differently

Example answer:

> I tried a modified attention block intended to improve long-context performance by adding a learned gate between local and global attention. The hypothesis was that the model would use local attention for syntax and global attention for long-range dependencies, improving long-context QA without a large compute increase.

Possible implementation:

```python
class GatedLocalGlobalAttention(nn.Module):
    def __init__(self, local_attn, global_attn, d_model):
        super().__init__()
        self.local_attn = local_attn
        self.global_attn = global_attn
        self.gate = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, C]
        local_out = self.local_attn(x)       # [B, T, C]
        global_out = self.global_attn(x)     # [B, T, C]

        gate = torch.sigmoid(self.gate(x))   # [B, T, 1]
        out = gate * global_out + (1 - gate) * local_out  # [B, T, C]
        return out                           # [B, T, C]
```

The project failed because:

- training was less stable than the baseline
- the gate saturated early
- long-context evals improved slightly, but short-context perplexity regressed
- inference became slower
- ablations showed most gains came from extra parameters, not the routing idea

Debugging the failure:

```python
def inspect_gate(model, input_ids):
    # input_ids: [B, T]
    gate_values = model.collect_gate_values(input_ids)  # [num_layers, B, T, 1]

    mean_gate_by_layer = gate_values.mean(dim=(1, 2, 3))  # [num_layers]
    std_gate_by_layer = gate_values.std(dim=(1, 2, 3))    # [num_layers]

    return mean_gate_by_layer, std_gate_by_layer
```

If `mean_gate_by_layer` is near 0 or 1 and `std_gate_by_layer` is tiny, the gate is not making nuanced routing decisions.

What I learned:

- A mechanism that sounds useful may not be used by the optimizer.
- Extra parameters can explain gains, so parameter-matched baselines matter.
- Long-context improvements are not valuable if they damage common short-context use.
- Instrumentation should be added early, not after the experiment fails.

How to present this in an interview:

> The project did not produce a publishable or shippable improvement, but it taught us that the bottleneck was not simply local-vs-global routing. The better next step was to simplify the method, add parameter-matched baselines, and evaluate long-context behavior earlier in training.

Interview answer:

> I would describe the failed hypothesis, the controlled experiment, the negative result, and the debugging process. The important part is showing that I learned something actionable rather than just saying the idea did not work.

Commentary: If you are asked this in a real interview, use a real project from your experience. The example above is a template, not something to claim as personal history unless it actually happened.

## 100. What Research Direction in LLMs Are You Most Excited About and Why?

One strong answer is:

> I am excited about improving reliable reasoning and verification in LLMs.

The reason is that current LLMs are powerful but still brittle. They can produce fluent answers that are wrong, and they often do not know when they are wrong. Better reasoning is not only about higher benchmark scores; it is about making models more useful for science, coding, education, agents, and decision support.

Promising directions include:

- verifier-guided reasoning
- process supervision
- tool-augmented reasoning
- test-time compute scaling
- self-correction with external feedback
- synthetic data generated by verifiable environments
- reasoning models that can inspect and revise intermediate work

Verifier-guided training example:

```python
def train_with_verifier(policy_model, verifier, prompts):
    # prompts: list[str], length B
    responses = policy_model.generate(
        prompts,
        num_return_sequences=K,
        temperature=0.7,
    )  # conceptual shape: [B, K] strings

    rewards = []
    for prompt_responses in responses:
        prompt_rewards = []
        for response in prompt_responses:
            reward = verifier(response)  # scalar, e.g. 1 if correct else 0
            prompt_rewards.append(reward)
        rewards.append(prompt_rewards)

    rewards = torch.tensor(rewards, dtype=torch.float32)  # [B, K]
    return rewards
```

For code, the verifier can be unit tests.

```python
def code_verifier(generated_code, tests):
    # generated_code: string
    # tests: callable test suite
    try:
        return float(tests(generated_code))  # scalar 0.0 or 1.0
    except Exception:
        return 0.0
```

For math, the verifier can be a symbolic checker when possible.

```python
def math_verifier(predicted_answer, gold_answer):
    # predicted_answer: scalar or expression
    # gold_answer: scalar or expression
    return float(equivalent(predicted_answer, gold_answer))  # scalar
```

The research question:

> How do we make models produce answers that are not only plausible, but checkable, correct, and robust?

A simple test-time compute pattern:

```python
def best_of_k_reasoning(model, verifier, prompt, K):
    candidates = model.generate(
        [prompt] * K,
        temperature=0.7,
        max_new_tokens=1024,
    )  # list[str], length K

    scores = torch.tensor([verifier(c) for c in candidates])  # [K]
    best_idx = scores.argmax()                                # scalar
    return candidates[int(best_idx)]                          # string
```

Why this direction matters:

- It connects model generation with objective feedback.
- It can improve reliability without relying only on larger pretraining runs.
- It is useful for domains where answers can be checked.
- It may help models learn better internal search and planning.

Risks and open questions:

- Verifiers can be wrong or reward-hackable.
- Some important tasks do not have easy verifiers.
- Test-time compute can be expensive.
- Models may learn to exploit benchmark checkers.
- Reasoning traces may look convincing without being faithful.

Interview answer:

> I am most excited about reliable reasoning through verification, tool use, and test-time compute. Scaling has made models broadly capable, but many high-value tasks require correctness, not just fluency. Research that connects generation with verifiable feedback could make LLMs much more useful for coding, math, science, and agentic workflows.

Commentary: This is an opinionated answer. Other strong answers could focus on efficient architectures, multimodal models, continual learning, interpretability, data quality, or alignment. The best interview answer is one you can defend with concrete experiments.
