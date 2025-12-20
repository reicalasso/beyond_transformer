# Memory-as-a-Service (MaaS)

Persistent, queryable memory for AI agents with hierarchical layers and dynamic routing.

## Quick Start

```bash
pip install -e ".[maas]"
python -m pulse.maas.server
```

## Python API

```python
from pulse.maas import MemoryService, MemoryLayer
import torch

memory = MemoryService(hidden_size=768)

memory_id = memory.write_memory(
    content="User prefers Python",
    embedding=torch.randn(768),
    layer=MemoryLayer.LONG_TERM,
    importance=0.9
)

results = memory.read_memory(
    query="user preferences",
    query_embedding=torch.randn(768),
    limit=5
)
```

## REST API

```bash
python -m pulse.maas.server

curl -X POST http://localhost:5000/pulse/memory/write \
  -H "Content-Type: application/json" \
  -d '{"content": "User loves AI", "type": "preference"}'

curl -X POST http://localhost:5000/pulse/memory/read \
  -H "Content-Type: application/json" \
  -d '{"query": "what does user love", "limit": 5}'
```

## Memory Layers

| Layer | Capacity | TTL | Use Case |
|-------|----------|-----|----------|
| Working | 32 | 1 hour | Active context |
| Short-term | 128 | 24 hours | Recent sessions |
| Long-term | 512 | Persistent | Important knowledge |

## API Endpoints

- `POST /pulse/memory/write` - Write memory
- `POST /pulse/memory/read` - Query memories
- `PUT /pulse/memory/update` - Update memory
- `DELETE /pulse/memory/delete` - Delete memory
- `POST /pulse/memory/consolidate` - Trigger consolidation
- `GET /pulse/memory/stats` - Get statistics
- `GET /pulse/health` - Health check

## Configuration

```python
memory = MemoryService(
    hidden_size=768,
    working_slots=32,
    short_term_slots=128,
    long_term_slots=512,
)
```
