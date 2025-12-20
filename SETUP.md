# 🚀 PULSE + MaaS Setup Guide

## Quick Install

```bash
# Clone or navigate to repository
cd beyond_transformer

# Install package
pip install -e .

# Install MaaS dependencies
pip install torch flask flask-cors
```

## Verify Installation

```bash
# Quick test
python test_maas_simple.py

# Run demo
python examples/maas_integration_demo.py
```

## Start Using

### Python API
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

### REST API Server
```bash
python -m pulse.maas.server
# Server runs on http://localhost:5000
```

## Documentation

- **Main README**: `README.md`
- **MaaS Guide**: `docs/MAAS.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **Examples**: `examples/`

## Troubleshooting

### Import Error
```bash
pip install -e .
```

### PyTorch Not Found
```bash
pip install torch
```

### Flask Not Found
```bash
pip install flask flask-cors
```

## Next Steps

1. Read `docs/MAAS.md` for complete documentation
2. Try examples in `examples/` directory
3. Run tests with `pytest tests/test_maas.py`
4. Check `docs/ARCHITECTURE.md` for technical details
