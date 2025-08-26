# Neural State Machine (NSM) Module

This module implements the core components of the Neural State Machine architecture.

## StatePropagator

The `StatePropagator` class implements gated updates for controlling the update, retain, or reset behavior of state vectors, inspired by LSTM/GRU architectures.

### Features

- **Gated Updates**: Implements both LSTM-style and GRU-style gating mechanisms
- **Differentiable**: Fully compatible with PyTorch's autograd
- **Flexible**: Supports different state dimensions and batch sizes

### Usage

```python
import torch
from nsm.state_propagator import StatePropagator

# Initialize with state dimension and gate type
propagator = StatePropagator(state_dim=128, gate_type='gru')  # or 'lstm'

# Create state and input tensors
prev_state = torch.randn(batch_size, state_dim)
new_input = torch.randn(batch_size, state_dim)

# Propagate state
updated_state = propagator(prev_state, new_input)
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

### Testing

Run the tests with:
```bash
python src/nsm/test_state_propagator.py
```