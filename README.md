# PULSE — Parallel Unified Linear State Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-orange.svg)]()

> A personal research project exploring whether a single, simple primitive can replace the SSM + Attention + State complexity of modern sequence models.

---

## What This Is

PULSE is an experimental sequence architecture built around one question:

**Can a uniform, O(n) block — combining local convolution and linear attention — match transformer-class models without quadratic complexity?**

This is not a production library. It's a focused research implementation where I'm testing architectural assumptions from scratch.

---

## Core Idea

Instead of stacking multiple specialized modules (self-attention, SSM, cross-attention, state banks), PULSE uses a single `UnifiedBlock`:

```
x → RMSNorm → [LocalConv ⊕ LinearAttn] → GatedFusion → + → RMSNorm → SwiGLU → +
```

- **LocalConv** — depthwise separable convolution for short-range patterns, O(n)
- **LinearAttention** — kernel-based causal attention with exponential decay, O(n)  
- **GatedFusion** — learned interpolation between local and global signals
- **KeyValueMemory** — optional fixed-size LRU cache for long-range recall

Every layer is identical. No conditionals, no special first/last layer logic.

---

## Current State

| Component | Status | Notes |
|---|---|---|
| `UnifiedBlock` | ✅ Working | LocalConv + LinearAttn + GatedFusion |
| `LinearAttention` | ⚠️ In progress | Correct but not yet vectorized — Python loop over sequence, needs associative scan |
| `KeyValueMemory` | ✅ Working | Circular buffer, O(1) write, O(k) read |
| `RecurrentState` | ✅ Working | Gated EMA-style update |
| `PulseForCausalLM` | ✅ Working | Full model with generation |
| Training script | ✅ Working | TinyStories, AdamW + cosine LR |
| Benchmarks | 🔲 Planned | No results yet — will add after LinearAttention vectotization |

**Known issues:**
- `LinearAttention.forward()` uses a Python `for` loop over the sequence dimension. This is O(n) in FLOPs but slow in practice. A proper associative scan (parallel prefix) is needed.
- Gate computation in `UnifiedBlock` calls `self.gate(combined)` twice — redundant forward pass, being fixed.

---

## Installation

```bash
git clone https://github.com/kaelvalen/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

## Quick Start

```python
from pulse import PulseConfig, PulseForCausalLM
import torch

config = PulseConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=8,
)

model = PulseForCausalLM(config)
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]

generated = model.generate(input_ids, max_new_tokens=100)
```

```bash
# Smoke test
python scripts/smoke_test.py

# Train on TinyStories
python scripts/train.py
```

---

## Architecture Decisions

### Why linear attention instead of softmax attention?

Softmax attention is O(n²) in sequence length. For long-context tasks this becomes a hard limit. Linear attention approximates attention with a kernel feature map (`elu(x) + 1`), reducing complexity to O(n) while maintaining a running KV summary across the sequence.

The tradeoff: weaker expressiveness on tasks that require precise token-to-token matching. The hypothesis being tested here is whether local convolution compensates for this.

### Why a single block type?

Conditional logic in architectures (different layers doing different things) makes ablations hard to interpret. A uniform block makes it easier to isolate what actually matters.

### Why not Mamba or RWKV?

I read both. PULSE is not an implementation of either — it's a different design informed by the same questions. The recurrent state here is deliberately simpler than Mamba's selective SSM.

---

## What's Next

- [ ] Vectorize `LinearAttention` with associative scan
- [ ] Fix gate double-computation in `UnifiedBlock`
- [ ] Run baseline benchmarks on TinyStories (loss, perplexity, sample quality)
- [ ] Ablation: with/without `KeyValueMemory`
- [ ] Ablation: with/without `RecurrentState`
- [ ] Compare against a vanilla transformer baseline at same parameter count

---

## File Structure

```
src/pulse/
├── core/
│   ├── unified.py      # UnifiedBlock, LinearAttention, LocalConv, RecurrentState
│   ├── memory.py       # KeyValueMemory, MemoryAugmentedLayer
│   ├── attention.py    # GQA, MHA (legacy)
│   ├── ffn.py          # SwiGLU
│   ├── norm.py         # RMSNorm
│   └── rope.py         # Rotary embeddings
└── models/
    ├── pulse_model.py  # PulseConfig, PulseModel, PulseForCausalLM
    ├── pulse_v2.py     # Explicit v2
    └── pulse_legacy.py # v1 compatibility

docs/
├── ARCHITECTURE.md     # Design decisions in detail
└── EXPERIMENTS.md      # Evaluation protocol
```

---

## What This Is Not

- Not a benchmark against GPT-4 or any production model
- Not a claim that linear attention is strictly better than softmax attention
- Not finished — active development, things will break

---

*Feedback, issues, and architecture critiques welcome.*
