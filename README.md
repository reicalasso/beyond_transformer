# PULSE 3 - Parallel Unified Linear State Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Keep it simple. Keep it efficient. Make it godlike.**

A radically simplified neural architecture with O(n) complexity.

## What's New in this version

- **UnifiedBlock**: Single primitive replacing SSM + Attention + State
- **LinearAttention**: O(n) attention with exponential decay
- **KeyValueMemory**: LRU-style cache replacing 3-tier hierarchical memory
- **~70% less code**, same or better performance

## Installation

```bash
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

## Quick Start

```python
from pulse import PulseConfig, PulseForCausalLM
import torch

# Minimal config - just what matters
config = PulseConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=8,
)

model = PulseForCausalLM(config)
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]

# Generation
generated = model.generate(input_ids, max_new_tokens=100)
```

## Core Components

```python
from pulse import UnifiedBlock, KeyValueMemory, LinearAttention

# UnifiedBlock: Local conv + linear attention + SwiGLU
block = UnifiedBlock(hidden_size=768, num_heads=8)
output, state = block(x)  # O(n) complexity

# KeyValueMemory: Fixed-size key–value cache
memory = KeyValueMemory(hidden_size=768, capacity=512)
memory.write(embedding)
values, scores, _ = memory.read(query, top_k=5)
```

## Architecture

```
src/pulse/
├── core/
│   ├── unified.py        # UnifiedBlock, LinearAttention, LocalConv
│   ├── memory.py         # KeyValueMemory, MemoryAugmentedLayer
│   ├── attention.py      # GQA, MHA (for legacy)
│   ├── ffn.py            # SwiGLU
│   ├── norm.py           # RMSNorm
│   └── rope.py           # Rotary embeddings
└── models/
    ├── pulse_model.py    # PulseConfig / PulseModel / PulseForCausalLM (current)
    ├── pulse_v2.py       # Explicit v2 implementation
    └── pulse_legacy.py   # v1 compatibility
```

## Design Philosophy

| Principle | Implementation |
|-----------|----------------|
| **One primitive** | UnifiedBlock does it all |
| **O(n) complexity** | LinearAttention + LocalConv |
| **Minimal state** | Single recurrent vector |
| **External memory** | Key–value cache, not hierarchical |
| **No conditionals** | Every layer identical |

## v1 → current PULSE Migration

```python
# v1 (legacy)
from pulse import LegacyPulseConfig, LegacyPulseForCausalLM

# Current PULSE model
from pulse import PulseConfig, PulseForCausalLM
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
