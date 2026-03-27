# PULSE v3 — Parallel Unified Linear State Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()

> A personal research project exploring whether a single, simple O(n) primitive can replace the SSM + Attention + State complexity of modern sequence models.

---

## Core Idea

PULSE v3 uses a single **DualStateBlock** repeated across all layers — no special first/last layers, no conditional logic:

```
x → RMSNorm → [LocalConv ⊕ DualStateAttention] → gate → residual
  → RMSNorm → SwiGLU FFN → residual
```

The key innovation is **dual-timescale linear attention**:

- **Fast state** (α ≈ 0.70, learnable): local syntax, short-range patterns
- **Slow state** (β ≈ 0.97, learnable): semantics, long-range context

Both run as O(n) causal decay scans. A learned gate blends the two timescales per position.

**LocalConv** — causal depthwise-separable convolution for sub-word patterns, O(n).

---

## Architecture

```
DualStateAttention:
  q, k, v  ←  qkv_proj(x)           # [B, H, T, D]
  q, k     ←  phi(·)                 # elu + 1, non-negative kernel
  kv       =  k * v                  # element-wise, [B, H, T, D]

  # Two independent decay scans
  fast_kv, fast_ks  ←  decay_scan(kv, k,  alpha)
  slow_kv, slow_ks  ←  decay_scan(kv, k,  beta)

  out_fast = (q * fast_kv) / (q · fast_ks)
  out_slow = (q * slow_kv) / (q · slow_ks)

  output  =  W_out( cat[out_fast, out_slow] )   # [B, T, D]
```

State per layer: `(kv_f, ks_f, kv_s, ks_s)` each `[B, H, D]` — 4D floats per head.

Incremental generation carries these states forward → **O(1) per decode step**.

---

## Quick Start

```bash
git clone https://github.com/kaelvalen/beyond_transformer.git
cd beyond_transformer
pip install -e ".[train]"
```

```python
from pulse import PulseConfig, PulseForCausalLM
import torch

config = PulseConfig(
    vocab_size=50257,
    hidden_size=512,
    num_layers=8,
    num_heads=8,
    fast_decay=0.70,
    slow_decay=0.97,
)

model = PulseForCausalLM(config)
out   = model(input_ids, labels=labels)
loss  = out["loss"]

generated = model.generate(input_ids, max_new_tokens=100, temperature=0.8)
```

```bash
# Sanity check
python scripts/smoke_test.py

# Train on TinyStories (defaults: 512-dim, 8-layer, 20k steps)
python scripts/train.py

# Train with RTX 4090 config (~50M params, 50k steps)
python scripts/train.py --config configs/rtx4090_tinystories.yaml

# Evaluate a checkpoint
python scripts/test_best_model.py --checkpoint output/pulse_v3_4090/best_model.pt
```

---

## Configuration

All hyperparameters are in `PulseConfig`:

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 512 | Model dimension |
| `num_layers` | 8 | Number of DualStateBlocks |
| `num_heads` | 8 | Attention heads |
| `ffn_mult` | 2.7 | FFN hidden = hidden × ffn_mult |
| `kernel_size` | 4 | LocalConv kernel size |
| `fast_decay` | 0.70 | Initial fast-state decay (learnable) |
| `slow_decay` | 0.97 | Initial slow-state decay (learnable) |
| `dropout` | 0.0 | Dropout probability |
| `tie_embeddings` | True | Tie input/output embeddings |

A YAML config can override any train.py CLI flag:

```yaml
# configs/my_config.yaml
hidden_size: 768
num_layers: 12
fast_decay: 0.70
slow_decay: 0.97
batch_size: 16
bf16: true
max_steps: 50000
output_dir: ./output/my_run
```

```bash
python scripts/train.py --config configs/my_config.yaml --lr 5e-4
```

---

## File Structure

```
src/pulse/
├── __init__.py     # exports PulseConfig, PulseModel, PulseForCausalLM
├── core.py         # RMSNorm, SwiGLU, _decay_scan, DualStateAttention, LocalConv, DualStateBlock
├── model.py        # PulseConfig, PulseModel, PulseForCausalLM
└── train.py        # self-contained training script with --config YAML support

scripts/
├── train.py        # thin launcher → src/pulse/train.py
├── smoke_test.py   # minimal forward + generate sanity check
└── test_best_model.py  # generation and perplexity evaluation

configs/
└── rtx4090_tinystories.yaml   # 768-dim, 12-layer, 50k steps on 24 GB GPU
```

---

## What's Next

- [ ] Benchmark vs. vanilla transformer at same parameter count on TinyStories
- [ ] Ablation: single vs. dual timescale
- [ ] Ablation: with/without LocalConv
- [ ] Replace Python chunk-loop in `_decay_scan` with parallel prefix scan (CUDA / `torch.compile`)

---

## What This Is Not

- Not a claim that linear attention beats softmax attention
- Not production-ready — active research, things will break

---

*Feedback, issues, and architecture critiques welcome.*
