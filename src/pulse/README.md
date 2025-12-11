# PULSE (PULSE) Module

This module implements the core components of the PULSE architecture.

## StatePropagator

The `StatePropagator` class implements gated updates for controlling the update, retain, or reset behavior of state vectors, inspired by LSTM/GRU architectures. It also supports state-to-state communication using MultiHeadAttention.

### Features

- **Gated Updates**: Implements both LSTM-style and GRU-style gating mechanisms
- **State-to-State Communication**: Allows state embeddings to communicate with each other using MultiHeadAttention
- **Differentiable**: Fully compatible with PyTorch's autograd
- **Flexible**: Supports different state dimensions, batch sizes, and numbers of states

### Usage

```python
import torch
from pulse.state_propagator import StatePropagator

# Initialize with state dimension and gate type
propagator = StatePropagator(
    state_dim=128, 
    gate_type='gru',  # or 'lstm'
    num_heads=4,      # for attention
    enable_communication=True  # enable state-to-state communication
)

# For single state
prev_state = torch.randn(batch_size, state_dim)
new_input = torch.randn(batch_size, state_dim)
updated_state = propagator(prev_state, new_input)

# For multiple states with communication
num_states = 8
prev_states = torch.randn(batch_size, num_states, state_dim)
new_inputs = torch.randn(batch_size, num_states, state_dim)
updated_states = propagator(prev_states, new_inputs)
```

### Gate Types

1. **GRU-style** (`gate_type='gru'`):
   - Reset gate: determines how much past state to forget
   - Update gate: determines how much new information to add
   - Uses the formula: `h_t = (1 - update_gate) * prev_state + update_gate * candidate_state`

2. **LSTM-style** (`gate_type='lstm'`):
   - Forget gate: determines what information to discard
   - Input gate: determines what new information to store
   - Output gate: determines what to output based on cell state

### State-to-State Communication

When enabled, the StatePropagator allows state nodes to attend to all other state nodes using MultiHeadAttention:
- Each state node attends to all other state nodes
- Optional residual connection and layer normalization
- Useful for relational reasoning between memory slots

### Testing

Run the tests with:
```bash
python src/pulse/test_state_propagator.py
```

## Hyperparameter Sweep

The notebook `notebooks/hyperparameter_sweep.ipynb` implements a sweep to test different numbers of state nodes (8, 16, 32, 64) across multiple datasets, recording:
- Accuracy
- Memory usage
- Training speed