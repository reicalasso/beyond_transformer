# PULSE - Parallel Unified Linear State Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A biologically-inspired neural architecture with hierarchical memory and efficient state-based processing.

## Installation

```bash
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

## Quick Start

### Core Model

```python
from pulse import PulseConfig, PulseForCausalLM
import torch

config = PulseConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=12,
)

model = PulseForCausalLM(config)
outputs = model(input_ids, labels=labels)
```

### Memory-as-a-Service (MaaS)

```python
from pulse.maas import MemoryService, MemoryLayer
import torch

memory = MemoryService(hidden_size=768)
memory_id = memory.write_memory(
    content="User prefers Python",
    embedding=torch.randn(768),
    layer=MemoryLayer.LONG_TERM
)

results = memory.read_memory(
    query="user preferences",
    query_embedding=torch.randn(768),
    limit=5
)
```

### MaaS Server

```bash
python -m pulse.maas.server
# Server runs on http://localhost:5000
```

## Training

```bash
python scripts/train.py --output-dir ./output --max-steps 2000
```

## Architecture

```
src/pulse/
├── core/              # Building blocks
│   ├── attention.py   # GQA, MHA, sparse attention
│   ├── cache.py       # KV cache variants
│   ├── ffn.py         # SwiGLU, MLP
│   ├── memory.py      # Hierarchical memory
│   ├── mixture.py     # MoE, MoD
│   ├── norm.py        # RMSNorm
│   ├── rope.py        # Rotary embeddings
│   ├── speculative.py # Speculative decoding
│   ├── spiking.py     # Pulse processing
│   ├── ssm.py         # State space models
│   └── state.py       # State management
├── maas/              # Memory-as-a-Service
│   ├── api.py
│   ├── consolidation.py
│   ├── memory_service.py
│   ├── query_engine.py
│   └── server.py
└── models/
    └── pulse.py       # Main model
```

## Key Features

- **Hierarchical Memory**: Working, short-term, and long-term layers with automatic consolidation
- **Efficient Attention**: Sparse and linear attention mechanisms
- **Dynamic Routing**: MoE-style expert selection
- **State Management**: GRU/LSTM-based state propagation
- **Memory Service**: Persistent, queryable memory for AI agents

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [MaaS Guide](docs/MAAS.md)
- [Examples](examples/)

## License

MIT License - see [LICENSE](LICENSE) file for details.
