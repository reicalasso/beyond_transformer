# 🔥 PULSE + MaaS - Memory-as-a-Service

## Quick Start

```bash
# Install
pip install -e .
pip install torch flask flask-cors

# Test
python test_maas_simple.py

# Run demo
python examples/maas_integration_demo.py

# Start server
python -m pulse.maas.server
```

## Python API

```python
from pulse.maas import MemoryService, MemoryLayer
import torch

# Initialize
memory = MemoryService(hidden_size=768)

# Write
memory_id = memory.write_memory(
    content="User prefers Python",
    embedding=torch.randn(768),
    layer=MemoryLayer.LONG_TERM,
    importance=0.9
)

# Read
results = memory.read_memory(
    query="what does user like",
    query_embedding=torch.randn(768),
    limit=5
)
```

## REST API

```bash
# Start server
python -m pulse.maas.server

# Write memory
curl -X POST http://localhost:5000/pulse/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "User loves AI", "type": "preference"}'

# Read memory
curl -X POST http://localhost:5000/pulse/memory/read \
  -H "Content-Type: application/json" \
  -d '{"query": "what does user love", "limit": 5}'
```

## Memory Layers

| Layer | Capacity | TTL | Compression | Use Case |
|-------|----------|-----|-------------|----------|
| **Working** | 32 slots | 1 hour | None (1.0x) | Active context |
| **Short-term** | 128 slots | 24 hours | 50% (0.5x) | Recent sessions |
| **Long-term** | 512 slots | Persistent | 75% (0.25x) | Important knowledge |

## Features

- **Hierarchical Memory**: 3-tier system with auto consolidation
- **Dynamic Routing**: 2-3x faster queries (smart layer selection)
- **Auto Consolidation**: Time/importance/access-based promotion
- **Natural Forgetting**: Decay and forgetting like human memory
- **Semantic Search**: Embedding-based retrieval
- **REST API**: Complete HTTP interface

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/pulse/memory/write` | POST | Write memory |
| `/pulse/memory/read` | POST | Query memories |
| `/pulse/memory/update` | PUT | Update memory |
| `/pulse/memory/delete` | DELETE | Delete memory |
| `/pulse/memory/consolidate` | POST | Trigger consolidation |
| `/pulse/memory/stats` | GET | Get statistics |
| `/pulse/health` | GET | Health check |

## Architecture

```
Working Memory (32 slots, 1h TTL)
    ↓ Consolidation
Short-Term Memory (128 slots, 24h TTL)
    ↓ Consolidation
Long-Term Memory (512 slots, persistent)
```

**Consolidation Rules:**
- Working → Short-term: age > 1h OR importance > 0.7
- Short-term → Long-term: age > 24h AND importance > 0.8
- Importance-based: High importance → immediate promotion
- Access-based: Frequent access → retention

## Examples

See `examples/` directory:
- `maas_basic_usage.py` - Basic operations
- `maas_advanced_usage.py` - Advanced features
- `maas_integration_demo.py` - Complete chatbot simulation
- `maas_server_example.py` - Server usage

## Configuration

```python
memory = MemoryService(
    hidden_size=768,
    working_slots=32,
    short_term_slots=128,
    long_term_slots=512,
    consolidation_threshold=0.7,
    decay_interval=100
)
```

## Performance

- **Query Speed**: 2-3x faster with dynamic routing
- **Memory Usage**: 66% reduction with compression
- **Complexity**: O(k) vs O(n) search

## Use Cases

- **Chatbots**: Remember user preferences and history
- **AI Assistants**: Maintain context across sessions
- **Knowledge Management**: Organize and retrieve information
- **Multi-Agent Systems**: Shared memory between agents

## More Information

- **Architecture Details**: See `docs/ARCHITECTURE.md`
- **Examples**: See `examples/` directory
- **Tests**: Run `python test_maas_simple.py` or `pytest tests/`

## License

MIT License
