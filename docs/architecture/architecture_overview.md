# Architecture Overview

This document provides a detailed overview of the PULSE (PULSE) architecture.

## Core Components

### 1. StatePropagator

The `StatePropagator` is the fundamental building block of PULSE that controls how state vectors are updated, retained, or reset.

#### Key Features:
- **Gated Updates**: Inspired by LSTM/GRU architectures with reset, update, and forget gates
- **State-to-State Communication**: Multi-head attention allowing state embeddings to communicate with each other
- **Differentiable**: Fully compatible with PyTorch's autograd
- **Flexible**: Supports different state dimensions, batch sizes, and numbers of states

#### Usage:
```python
import torch
from pulse import StatePropagator

# Initialize with state dimension and gate type
propagator = StatePropagator(
    state_dim=128, 
    gate_type='gru',  # or 'lstm'
    num_heads=4,      # for attention
    enable_communication=True  # enable state-to-state communication
)

# For single state update
batch_size = 32
prev_state = torch.randn(batch_size, 128)
new_input = torch.randn(batch_size, 128)
updated_state = propagator(prev_state, new_input)

# For multiple states with communication
num_states = 16
prev_states = torch.randn(batch_size, num_states, 128)
new_inputs = torch.randn(batch_size, num_states, 128)
updated_states = propagator(prev_states, new_inputs)
```

### 2. TokenToStateRouter

Routes input tokens to appropriate state nodes based on learned routing mechanisms.

#### Key Features:
- **Multi-head Attention**: Routes tokens using learned attention mechanisms
- **Content-based Routing**: Tokens are routed based on semantic similarity to states
- **Flexible Dimensions**: Handles different token and state dimensions

### 3. StateManager

Manages state nodes with learnable importance scores and dynamic pruning.

#### Key Features:
- **Dynamic Allocation**: Allocates new state nodes when needed
- **Importance Scoring**: Learnable importance scores for each state node
- **Automatic Pruning**: Prunes low-importance states based on threshold

### 4. PulseLayer

Complete PULSE layer combining token-to-state routing with state propagation.

## Architecture Design

### State Update Mechanism

The PULSE uses gated mechanisms similar to RNNs but with explicit state management:

1. **Reset Gate**: Determines how much past state to forget
2. **Update Gate**: Determines how much new information to add
3. **Candidate State**: Computes potential new state values

The update formula follows the GRU pattern:
```
h_t = (1 - update_gate) * prev_state + update_gate * candidate_state
```

### State-to-State Communication

States communicate with each other using MultiHeadAttention:
- Each state node attends to all other state nodes
- Optional residual connection and layer normalization
- Useful for relational reasoning between memory slots

### Token-to-State Routing

Tokens are routed to states using attention mechanisms:
- Tokens attend to relevant states based on learned routing
- Routing weights determine token-state assignment
- Enables efficient information flow from tokens to states

## Mathematical Formulation

### Gated State Update

For GRU-style gating:
```
r_t = σ(W_r · [h_{t-1}, x_t])
z_t = σ(W_z · [h_{t-1}, x_t])
ĥ_t = tanh(W_h · [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * ĥ_t
```

Where:
- `r_t`: Reset gate
- `z_t`: Update gate  
- `ĥ_t`: Candidate state
- `h_t`: Updated state
- `σ`: Sigmoid activation
- `*`: Element-wise multiplication

### Attention-based Routing

Token-to-state routing uses scaled dot-product attention:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- `Q`: Query (states)
- `K`: Key (tokens)  
- `V`: Value (tokens)
- `d_k`: Key dimension

## Performance Characteristics

### Time Complexity
- **Token Processing**: O(n·s) where n = sequence length, s = number of states
- **State Communication**: O(s²) for attention between states
- **Overall**: O(n·s + s²) vs O(n²) for Transformers

### Memory Complexity
- **State Storage**: O(s·d) where d = state dimension
- **Attention Weights**: O(s²) for state-to-state communication
- **Overall**: O(s·d + s²) vs O(n²) for attention weights in Transformers

### Scalability
- **Sequence Length**: Linear scaling with sequence length
- **State Count**: Linear scaling with state count (s << n typically)
- **Parallelization**: Token processing highly parallelizable

## Advantages Over Transformers

### 1. Computational Efficiency
- **Linear Time Complexity**: O(n·s) instead of O(n²)
- **Reduced Memory Usage**: No need to store full attention matrices
- **Scalable to Long Sequences**: Efficient processing of very long sequences

### 2. Interpretability
- **Explicit State Management**: Clear view of state evolution over time
- **Learnable Routing**: Understandable token-to-state assignment
- **Importance Scoring**: Quantifiable state importance

### 3. Dynamic Adaptation
- **Dynamic State Allocation**: Model complexity adapts to task requirements
- **Automatic Pruning**: Unnecessary states removed automatically
- **Flexible Architecture**: State count can vary during processing

## Implementation Details

### PyTorch Integration
All components are implemented as PyTorch `nn.Module` subclasses:
- Full autograd support for gradient computation
- GPU acceleration with CUDA tensors
- Integration with PyTorch ecosystem (optimizers, schedulers, etc.)

### State Management
States are managed as learnable parameters:
- `nn.Parameter` for persistent state storage
- Gradient-based optimization of state values
- Buffer registration for non-trainable state metadata

### Attention Mechanisms
Multi-head attention is implemented using PyTorch's built-in modules:
- `nn.MultiheadAttention` for efficient computation
- Support for different attention patterns
- Flexible configuration of heads and dimensions

## Use Cases

### 1. Long Sequence Modeling
- Efficient processing of documents, books, code
- Linear scaling with sequence length
- Persistent state for long-term dependencies

### 2. Memory-Intensive Tasks
- Tasks requiring explicit memory management
- Question answering with context retention
- Multi-step reasoning problems

### 3. Interpretability-Critical Applications
- Medical diagnosis systems
- Financial analysis tools
- Scientific computing applications

## Limitations and Future Work

### Current Limitations
- **State Count Selection**: Optimal state count requires empirical tuning
- **Initialization Sensitivity**: State initialization affects convergence
- **Limited Parallelization**: State-to-state communication creates dependencies

### Future Improvements
- **Adaptive State Count**: Learn optimal state count during training
- **Hierarchical States**: Multi-level state organization
- **Sparse Communication**: Efficient state-to-state attention patterns

This architecture overview provides the foundation for understanding how PULSEs process information and maintain state, offering a compelling alternative to traditional Transformer architectures.