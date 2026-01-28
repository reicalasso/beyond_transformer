# PULSE Architecture

## Design Philosophy

**Keep it simple. Keep it efficient.**

PULSE radically simplifies the architecture:

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

Causal, kernel-based attention layer with exponential decay:

- Computes queries, keys, values per head
- Applies non-negative feature map `elu(x) + 1`
- Maintains a lightweight running summary per head
- Applies an exponential decay factor to favor recent tokens

This implementation is designed to be:

- **O(n)** in sequence length
- **Mask-aware** (padding tokens do not affect the state)
- **Mixed-precision friendly** (no large intermediate matrices)

### KeyValueMemory

Fixed-size key–value cache replacing 3-tier memory:

```
┌─────────────────────────┐
│  KeyValueMemory (64)    │
│  ├─ keys[capacity, dim] │
│  ├─ values[capacity]    │
│  └─ ptr (circular)      │
└─────────────────────────┘
```

Operations:

- `write(key, value)`: O(1) circular buffer insert
- `read(query, top_k)`: O(k) similarity search

### RecurrentState

Single compressed state vector shared across layers and sequences.

- Input: current hidden states `[batch, seq_len, hidden_dim]`
- Output: updated state `[batch, hidden_dim]`
- Update rule: gated EMA-style blend of previous state and pooled hidden states

## Model Architecture

```
Input → Embed → [UnifiedBlock × N] → Norm → LM Head → Output
                     │
                     └─ RecurrentState (single vector, updated after all blocks)
                     └─ Optional: KeyValueMemory (LRU-style cache)
```

## File Structure

```
src/pulse/core/
├── unified.py        # UnifiedBlock, LinearAttention, LocalConv, RecurrentState
├── memory.py         # KeyValueMemory, MemoryAugmentedLayer
├── attention.py      # GQA, MHA (legacy compatibility)
├── ffn.py            # SwiGLU
├── norm.py           # RMSNorm
└── rope.py           # Rotary embeddings

src/pulse/models/
├── pulse_model.py    # PulseConfig, PulseModel, PulseForCausalLM (current)
├── pulse_v2.py       # explicit v2 implementation
└── pulse_legacy.py   # v1 compatibility
```

## Complexity Comparison

| Component     | v1                 | v2               |
| ------------- | ------------------ | ---------------- |
| Attention     | O(n²) GQA         | O(n) LinearAttn  |
| State         | 32 slots           | 1 vector         |
| Memory        | 3 tiers, 672 slots | 1 LRU, 512 slots |
| Layer types   | 4+ conditional     | 1 uniform        |
| Config params | ~20                | ~10              |

## Performance

**Sequence Processing:**

- Linear attention: O(n) vs O(n²)
- Local conv: O(n) with small kernel
- Total per layer: O(n)

**Memory:**

- KeyValueMemory: capacity × dim × 2 (keys + values), capacity typically 64
- RecurrentState: 1 × dim
