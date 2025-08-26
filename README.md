# Beyond Transformer: Neural State Machines

This repository explores Neural State Machines (NSM) as an alternative to traditional transformer architectures. NSMs maintain and update explicit state vectors, enabling more interpretable and potentially more efficient sequence processing.

## Features

### 1. Gated State Updates
- LSTM/GRU-inspired gates to control state update, retention, and reset behavior
- Configurable gating mechanisms ('lstm' or 'gru')

### 2. State-to-State Communication
- Multi-head attention allowing states to communicate with each other
- Optional residual connections and layer normalization
- Useful for relational reasoning between memory slots

### 3. Hyperparameter Analysis
- Comprehensive experiment testing different state counts (8, 16, 32, 64)
- Performance metrics: accuracy, memory usage, training speed

## Project Structure

```
src/
├── nsm/
│   ├── __init__.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── state_propagator.py      # Core state propagation logic
│   │   └── test_state_propagator.py # Unit tests
│   ├── models/
│   │   ├── __init__.py
│   │   └── simple_nsm.py            # Example NSM model
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── state_count_sweep.py     # Hyperparameter experiments
│   └── utils/
│       └── __init__.py
├── run_experiments.py               # Main experiment runner
└── plot_results.py                  # Results visualization
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd beyond_transformer

# No additional dependencies required beyond PyTorch
```

## Usage

### Basic Usage

```python
import torch
from nsm.modules import StatePropagator

# Initialize state propagator
propagator = StatePropagator(
    state_dim=128,
    gate_type='gru',              # or 'lstm'
    enable_communication=True      # Enable state-to-state communication
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

### Complete Model Example

```python
from nsm.models import SimpleNSM

# Create a simple NSM model
model = SimpleNSM(
    input_dim=784,      # Input dimension (e.g., flattened MNIST)
    state_dim=128,      # State vector dimension
    num_states=16,      # Number of state nodes
    output_dim=10,      # Output dimension (e.g., classification)
    gate_type='gru'     # Gating mechanism
)

# Forward pass
batch_size = 32
x = torch.randn(batch_size, 784)
output = model(x)
```

## Experiments

### State Count Hyperparameter Sweep

Run the experiment to test different numbers of state nodes:

```bash
python run_experiments.py
```

Results are saved to `hyperparameter_sweep_results.json` with metrics:
- Training accuracy
- Test accuracy
- Memory usage
- Training time

### Results Summary

Key findings from the state count experiment:
- Training accuracy varies with state count (53-75%)
- Test accuracy remains relatively stable (10-12%)
- Memory usage increases with more states
- Training time scales linearly with state count

## Testing

Run the unit tests:

```bash
python src/nsm/modules/test_state_propagator.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.