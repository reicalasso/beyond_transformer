# PULSE Architecture

## Core Components

### PULSE Model

```
Input → Token Embeddings → PULSE Layers → Output
                              │
                              ├─ Attention (GQA/MHA)
                              ├─ State Management
                              ├─ FFN (SwiGLU)
                              └─ Hierarchical Memory
```

**Key Modules:**
- `attention.py` - Grouped Query Attention, Multi-Head Attention
- `state.py` - State propagation and management
- `memory.py` - Hierarchical memory with consolidation
- `ffn.py` - Feed-forward networks (SwiGLU, MLP)
- `rope.py` - Rotary position embeddings
- `cache.py` - KV cache variants
- `mixture.py` - Mixture of Experts (MoE)
- `ssm.py` - State Space Models

### Memory-as-a-Service (MaaS)

```
API Layer (REST/Python)
    │
    ├─ MemoryService
    │   ├─ Memory Store (3 layers)
    │   ├─ Query Engine (dynamic routing)
    │   └─ Consolidator (time/importance-based)
    │
    └─ PULSE Core (HierarchicalMemory, MemoryBank)
```

**Memory Layers:**

```
Working (32 slots, 1h TTL)
    ↓ Consolidation
Short-term (128 slots, 24h TTL)
    ↓ Consolidation
Long-term (512 slots, persistent)
```

**Consolidation Rules:**
- Working → Short-term: age > 1h OR importance > 0.7
- Short-term → Long-term: age > 24h AND importance > 0.8

## Data Flow

### Write Operation
1. Convert content to embedding vector
2. Calculate importance score
3. Assign to layer (importance >= 0.8 → Long-term, >= 0.5 → Short-term, < 0.5 → Working)
4. Create MemoryEntry with metadata
5. Store in memory_store and update layer_indices
6. Return memory ID

### Read Operation
1. Convert query to embedding vector
2. Dynamic routing - determine which layers to query
3. Multi-layer search with similarity scoring
4. Aggregate and rank results
5. Update access patterns
6. Return top-k matches

### Consolidation Process

**Time-Based:**
- Working → Short-term: age > 1h OR importance > 0.7
- Short-term → Long-term: age > 24h AND importance > 0.8

**Importance-Based:**
- Working → Long-term: importance > 0.9
- Short-term → Long-term: importance > 0.95

**Access-Based:**
- Promote frequently accessed memories
- Boost importance for high access count

**Decay & Forgetting:**
- importance = importance × (0.99 ^ age_hours)
- Delete if importance < 0.05 AND unused > 24h

## Performance

**Query Speed:**
- Dynamic routing: 2-3x faster (search ~160 slots vs 672 slots)
- Complexity: O(k) vs O(n) with smart layer selection

**Memory Usage:**
- Working: 32 slots × 768 dim = ~96 KB
- Short-term: 128 slots × 384 dim = ~192 KB (50% compression)
- Long-term: 512 slots × 192 dim = ~384 KB (75% compression)
- Total: ~672 KB (66% reduction vs full precision)
