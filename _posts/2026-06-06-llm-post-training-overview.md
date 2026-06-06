---
title: "LLM Post-Training (1): Overview and Roadmap"
date: 2026-06-06 12:00:00 +0000
series_order: 1
categories: [fundamentals]
tags: [llm, post-training, sft, alignment, rlhf]
pin: true
math: false
permalink: /posts/llm-post-training-overview/
---

## Why This Tutorial Exists

If you are starting with LLM post-training, the first thing to understand is what happens **after** pre-training: how a base model becomes useful, steerable, and aligned with human intent.

This tutorial series will cover the full post-training pipeline — from supervised fine-tuning (SFT) through preference optimization (DPO, RLHF) — with intuition-first explanations and hands-on notes.

## The One-Line Mental Model

Most modern LLM post-training follows this recipe:

`base model -> SFT -> preference optimization -> evaluation -> deployment`

That is the whole pipeline in one line. The upcoming posts will explain each stage in detail.

## What Comes Next

Planned topics for this series:

1. **Pre-training vs post-training** — what each stage optimizes for
2. **Instruction tuning and SFT** — turning a base model into a helpful assistant
3. **Data curation** — formatting, quality filtering, and dataset design
4. **Preference optimization** — DPO, RLHF, and related methods
5. **Evaluation** — benchmarks, human eval, and failure analysis
6. **PEFT and efficient fine-tuning** — LoRA, QLoRA, and when to use them

Stay tuned — more posts coming soon.
