# PULSE v2 Architecture

## Design Philosophy

**Keep it simple. Keep it efficient.**

v2 radically simplifies the architecture:
- One primitive (UnifiedBlock) replaces SSM + Attention + State
- O(n) complexity everywhere
- Single recurrent state instead of state banks
- LRU cache instead of hierarchical memory

## Core Components

### UnifiedBlock

```
x → RMSNorm → [LocalConv ⊕ LinearAttn] → Gate → + → RMSNorm → SwiGLU → + → out
```

**Combines:**
- `LocalConv`: Depthwise convolution for local patterns (O(n))
- `LinearAttention`: Kernel attention with decay for global context (O(n))
- `SwiGLU`: Gated FFN

### LinearAttention

O(n) attention using kernel trick:
```python
# Instead of: softmax(QK^T)V  -- O(n²)
# We do: Q @ cumsum(K^T @ V)  -- O(n)
```

Features:
- Exponential decay for recency bias
- Cumulative state for streaming
- Feature map: elu(x) + 1

### SimpleMemory

Fixed-size LRU cache replacing 3-tier memory:
```
┌─────────────────────────┐
│  SimpleMemory (512)     │
│  ├─ keys[capacity, dim] │
│  ├─ values[capacity]    │
│  └─ ptr (circular)      │
└─────────────────────────┘
```

Operations:
- `write(key, value)`: O(1) circular buffer insert
- `read(query, top_k)`: O(k) similarity search

### RecurrentState

Single compressed state vector:
```python
# Old: [batch, 32, hidden_dim] state bank
# New: [batch, hidden_dim] single state

state = gate * state + (1 - gate) * update
```

## Model Architecture

```
Input → Embed → [UnifiedBlock × N] → Norm → LM Head → Output
                      │
                      └─ Optional: RecurrentState (single vector)
                      └─ Optional: SimpleMemory (LRU cache)
```

## File Structure

```
src/pulse/core/
├── unified.py        # UnifiedBlock, LinearAttention, LocalConv, RecurrentState
├── simple_memory.py  # SimpleMemory, MemoryAugmentedBlock
├── attention.py      # GQA, MHA (legacy compatibility)
├── ffn.py            # SwiGLU
├── norm.py           # RMSNorm
└── rope.py           # Rotary embeddings

src/pulse/models/
├── pulse_v2.py       # PulseV2Config, PulseV2, PulseV2ForCausalLM
└── pulse_legacy.py   # v1 compatibility
```

## Complexity Comparison

| Component | v1 | v2 |
|-----------|----|----|
| Attention | O(n²) GQA | O(n) LinearAttn |
| State | 32 slots | 1 vector |
| Memory | 3 tiers, 672 slots | 1 LRU, 512 slots |
| Layer types | 4+ conditional | 1 uniform |
| Config params | ~20 | ~10 |

## Performance

**Sequence Processing:**
- Linear attention: O(n) vs O(n²)
- Local conv: O(n) with small kernel
- Total per layer: O(n)

**Memory:**
- SimpleMemory: 512 × dim × 2 (keys + values)
- RecurrentState: 1 × dim
- ~50% reduction vs v1
