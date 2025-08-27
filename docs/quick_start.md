# Quick Start Guide

Get up and running with Neural State Machines in minutes!

## 🚀 5-Minute Quick Start

### 1. Installation

```bash
# Clone and install
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

### 2. Basic Usage

```python
import torch
from nsm import StatePropagator

# Create a state propagator
propagator = StatePropagator(state_dim=128, gate_type='gru')

# Process some data
batch_size = 32
prev_state = torch.randn(batch_size, 128)
new_input = torch.randn(batch_size, 128)
updated_state = propagator(prev_state, new_input)

print(f"State updated! Shape: {updated_state.shape}")
```

### 3. Complete Model

```python
from nsm.models import SimpleNSM

# Create NSM model
model = SimpleNSM(
    input_dim=784,      # e.g., flattened MNIST
    state_dim=128,
    num_states=16,
    output_dim=10,      # e.g., 10-class classification
    gate_type='gru'
)

# Forward pass
x = torch.randn(32, 784)
output = model(x)
print(f"Output shape: {output.shape}")
```

## 🎯 Common Use Cases

### 1. Sequence Modeling

```python
import torch
from nsm import NSMLayer

# Process sequences with NSM
nsm_layer = NSMLayer(
    state_dim=128,
    token_dim=64,
    num_heads=4
)

# Process sequence
batch_size, seq_len, token_dim = 32, 100, 64
tokens = torch.randn(batch_size, seq_len, token_dim)

# Initialize states
states = torch.randn(batch_size, 16, 128)

# Process sequence
for t in range(seq_len):
    token_t = tokens[:, t, :]  # [batch_size, token_dim]
    states = nsm_layer(states, token_t.unsqueeze(1).repeat(1, 16, 1))
```

### 2. Dynamic State Management

```python
from nsm import StateManager

# Create state manager with dynamic allocation
state_manager = StateManager(
    state_dim=128,
    max_states=64,
    initial_states=16,
    prune_threshold=0.3
)

# Get current states
states = state_manager()

# Dynamically manage states during training
state_manager.allocate_states(4)  # Add 4 new states
state_manager.prune_low_importance_states()  # Remove low-importance states
```

## 🧪 Running Your First Experiment

```bash
# Run a simple training experiment
python scripts/train_model.py --config configs/default_config.json

# Or try a quick debug run
python scripts/train_model.py --config configs/debug_config.yaml
```

## 📚 Next Steps

1. **Dive Deeper**: Check out the [Installation and Usage Guide](installation_and_usage.md)
2. **Explore Examples**: Look at notebooks in the `notebooks/` directory
3. **Run Experiments**: Try scripts in `src/nsm/experiments/`
4. **Read Documentation**: Browse detailed docs in `docs/`

## 🤝 Need Help?

- Check the [FAQ](faq.md) for common questions
- Open an [issue](https://github.com/yourusername/beyond_transformer/issues) for bugs or feature requests
- Join our [discussions](https://github.com/yourusername/beyond_transformer/discussions) for community support

Ready to explore Neural State Machines? [Start experimenting!](../notebooks/getting_started.ipynb)