# PULSE Experiment Protocol

This document describes a minimal, repeatable protocol for evaluating
changes to the PULSE architecture.

## Goals

- Keep the **core architecture stable and simple**
- Measure changes using **small, fast experiments**
- Focus on **failure modes** (repetition, subject drift, incoherent dialog)

## Default setup (TinyStories small)

- Dataset: `roneneldan/TinyStories`
- Tokenizer: `gpt2` (with `pad_token = eos_token`)
- Sequence length: `max_seq_len = 256`
- Model size:
  - `hidden_size = 512`
  - `num_layers = 4`
  - `num_heads = 8`
- Optimizer: AdamW, `lr = 5e-4`, cosine schedule with warmup
- Steps: `max_steps = 2000`, `eval_interval = 500`
- Mixed precision: FP16

These settings match `scripts/train.py` defaults when no YAML config is
provided and CLI args are used.

## Recommended ablations

For each architectural change, run at least the following experiments:

### 1. Baseline (no memory)

- Config:
  - `use_recurrent_state = True`
  - `use_memory = False`
- Purpose:
  - Measure the effect of recurrent state alone on:
    - validation loss / perplexity
    - repetition patterns
    - subject continuity

### 2. Memory (small capacity)

- Config:
  - `use_recurrent_state = True`
  - `use_memory = True`
  - `memory_capacity = 64`
- Purpose:
  - Measure whether a small external key–value cache improves
    long-range recall without causing echo/repetition.

### 3. Optional: larger models

Once the behavior is well-understood at the small scale, repeat with a
larger configuration (e.g. `hidden_size=768`, `num_layers=8`) to check
that the same qualitative behaviors hold.

## What to look at

For each run:

- **Metrics**
  - Best validation loss and perplexity
  - Training/validation gap
- **Text samples**
  - Prompts:
    - `"Once upon a time"`
    - `"The little girl"`
    - `"Tom was a boy who"`
    - `"In the forest"`
  - Check for:
    - token repetition (`the the the`, `girl girl girl`)
    - subject drift within and across sentences
    - dialog fluency vs. syntax errors

The goal is not to optimize TinyStories itself, but to use it as a
fast, informative probe of architectural changes to PULSE.

