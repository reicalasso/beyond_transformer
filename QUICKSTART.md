# 🚀 Quick Start

## 1. Install

```bash
cd beyond_transformer
pip install -e .
pip install torch flask flask-cors
```

## 2. Test

```bash
python test_maas_simple.py
```

## 3. Try Examples

```bash
# Basic usage
python examples/maas_basic_usage.py

# Complete demo
python examples/maas_integration_demo.py
```

## 4. Start Server

```bash
python -m pulse.maas.server
# Server runs on http://localhost:5000
```

## 5. Use API

```python
from pulse.maas import MemoryService, MemoryLayer
import torch

memory = MemoryService(hidden_size=768)
memory_id = memory.write_memory(
    content="User prefers Python",
    embedding=torch.randn(768),
    layer=MemoryLayer.LONG_TERM
)
```

## Next Steps

- **Documentation**: [docs/MAAS.md](docs/MAAS.md)
- **Examples**: [examples/](examples/)
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
