# PULSE v2 - Parallel Unified Linear State Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Keep it simple. Keep it efficient. Make it godlike.**

A radically simplified neural architecture with O(n) complexity.

## What's New in v2

- **UnifiedBlock**: Single primitive replacing SSM + Attention + State
- **LinearAttention**: O(n) attention with exponential decay
- **SimpleMemory**: LRU cache replacing 3-tier hierarchical memory
- **~70% less code**, same or better performance

## Installation

```bash
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

## Quick Start

```python
from pulse import PulseV2Config, PulseV2ForCausalLM
import torch

# Minimal config - just what matters
config = PulseV2Config(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=8,
)

model = PulseV2ForCausalLM(config)
outputs = model(input_ids, labels=labels)
loss = outputs["loss"]

# Generation
generated = model.generate(input_ids, max_new_tokens=100)
```

## Core Components

```python
from pulse import UnifiedBlock, SimpleMemory, LinearAttention

# UnifiedBlock: Local conv + Linear attention + SwiGLU
block = UnifiedBlock(hidden_size=768, num_heads=8)
output, state = block(x)  # O(n) complexity

# SimpleMemory: Fixed-size LRU cache
memory = SimpleMemory(hidden_size=768, capacity=512)
memory.write(embedding)
values, scores, _ = memory.read(query, top_k=5)
```

## Architecture

```
src/pulse/
├── core/
│   ├── unified.py        # UnifiedBlock, LinearAttention, LocalConv
│   ├── simple_memory.py  # SimpleMemory, MemoryAugmentedBlock
│   ├── attention.py      # GQA, MHA (for legacy)
│   ├── ffn.py            # SwiGLU
│   ├── norm.py           # RMSNorm
│   └── rope.py           # Rotary embeddings
└── models/
    ├── pulse_v2.py       # PulseV2 (recommended)
    └── pulse_legacy.py   # PulseV1 (compatibility)
```

## Design Philosophy

| Principle | Implementation |
|-----------|----------------|
| **One primitive** | UnifiedBlock does it all |
| **O(n) complexity** | LinearAttention + LocalConv |
| **Minimal state** | Single recurrent vector |
| **Simple memory** | LRU cache, not hierarchical |
| **No conditionals** | Every layer identical |

## v1 → v2 Migration

```python
# v1 (legacy)
from pulse import PulseConfig, PulseForCausalLM

# v2 (recommended)  
from pulse import PulseV2Config, PulseV2ForCausalLM
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
