# Beyond Transformer: Neural State Machines

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

A research project exploring Neural State Machines (NSM) as an alternative to traditional transformer architectures. NSMs maintain and update explicit state vectors, enabling more interpretable and potentially more efficient sequence processing.

## Overview

This repository implements Neural State Machines (NSM), a novel approach to sequence modeling that combines the strengths of recurrent models (state, memory) with Transformers (parallel attention, scalability) to create a more efficient and powerful architecture for the future of AI.

### Key Features

1. **Gated State Updates**: LSTM/GRU-inspired gates to control state update, retention, and reset behavior
2. **State-to-State Communication**: Multi-head attention allowing states to communicate with each other
3. **Dynamic State Allocation and Pruning**: Learnable importance scores for each state node with automatic pruning
4. **Hybrid Attention Mechanisms**: Token-to-state routing with learned attention and content-based attention
5. **Interpretability**: Explicit state management providing better understanding of model decisions
6. **Efficiency**: O(n·s) complexity (s = number of states ≪ n) instead of O(n²) attention

## Architecture

The Neural State Machine consists of several core components:

- **TokenToStateRouter**: Routes input tokens to appropriate state nodes based on learned attention mechanisms
- **StateManager**: Manages state nodes with learnable importance scores and dynamic allocation/pruning
- **StatePropagator**: Controls state updates using gating mechanisms and enables state-to-state communication
- **HybridAttention**: Combines token-to-state routing with content-based attention for information flow

## Performance Characteristics

| Architecture | Performance | Memory Usage | Training Time | Inference Time | Interpretability |
|--------------|-------------|--------------|---------------|----------------|------------------|
| Transformer (Baseline) | High | High (O(n²)) | High | High (O(n²)) | Medium |
| Efficient Transformers | Medium-High | Medium | Medium | Medium | Medium |
| RWKV | High | Low | Medium | Low | Medium |
| Mamba/S4 | High | Low | Medium | Low | Low-Medium |
| **NSM (Proposed)** | High | Low (O(s)) | Medium | Medium-Low | High |

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer

# Install dependencies
pip install -r requirements/requirements.txt

# Install the package in development mode
pip install -e .
```

### Using conda

```bash
# Create and activate conda environment
conda env update -f environment.yml
conda activate beyond_transformer

# Install the package
pip install -e .
```

### Development Installation

For development and running experiments:

```bash
# Install development dependencies
pip install -r requirements/requirements-experiments.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from nsm import StatePropagator, NSMLayer, StateManager

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

### Advanced Usage with Dynamic State Management

```python
from nsm import NSMLayer, StateManager

# Create state manager with dynamic allocation
state_manager = StateManager(
    state_dim=128,
    max_states=64,
    initial_states=16,
    prune_threshold=0.3
)

# Create NSM layer
nsm_layer = NSMLayer(state_dim=128, token_dim=64, num_heads=4)

# Get current states
states = state_manager()

# During training, periodically prune and allocate states
pruned_count = state_manager.prune_low_importance_states()
allocated_count = state_manager.allocate_states(2)
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

## Running Experiments

### Configuration

The project uses configuration files to manage model parameters, training settings, and experiment details. Configuration files are available in both JSON and YAML formats in the `configs/` directory.

### Available Configurations

- `configs/default_config.json`: Default configuration
- `configs/small_model_config.json`: Configuration for small models
- `configs/large_model_config.json`: Configuration for large models
- `configs/long_training_config.json`: Configuration for long training experiments
- `configs/debug_config.yaml`: Debug configuration in YAML format

### Running Experiments

To use a configuration file in training:

```bash
python scripts/train_model.py --config configs/default_config.json
```

## Key Research Findings

### State Count Analysis
- Training accuracy varies with state count (53-75% in experiments)
- Test accuracy remains relatively stable (10-12% for synthetic data)
- Memory usage increases linearly with state count
- Training time scales with state count

### Dynamic State Management
- Automatic pruning can reduce memory footprint
- State importance scores provide interpretability
- Dynamic allocation allows adaptive model complexity

## Project Structure

```
.
├── .github/
│   └── workflows/              # CI/CD workflows
├── configs/                    # Configuration files
├── data/                       # Data directory (not committed)
├── docs/                       # Documentation
├── notebooks/                   # Jupyter notebooks
├── references/                  # Reference materials
├── requirements/               # Dependency files
│   ├── requirements.txt
│   ├── requirements-experiments.txt
│   └── requirements-test.txt
├── results/                     # Experiment results
├── scripts/                     # Utility scripts
├── src/                         # Source code
│   └── nsm/                     # Neural State Machine implementation
└── tests/                       # Test files
```

## Testing

### Unit Tests

Run comprehensive unit tests for all components:

```bash
# Run all tests with pytest
python -m pytest tests/ -v
```

### Core Module Tests

```bash
# Legacy module tests
python src/nsm/modules/test_state_propagator.py

# Component integration tests
python src/nsm/test_components.py
```

### Test Coverage

The test suite includes:
- **Shape verification** for all tensor operations
- **Differentiability testing** for gradient flow
- **Probability constraints** (softmax outputs sum to 1)
- **Edge case handling** (boundary conditions, error cases)
- **Integration testing** between components
- **Performance smoke tests** for basic functionality

## Documentation

Detailed documentation is available in the `docs/` directory:

- [`docs/architecture_overview.md`](docs/architecture_overview.md): Detailed architecture description
- [`docs/component_reference.md`](docs/component_reference.md): API reference for all components
- [`docs/experiments_guide.md`](docs/experiments_guide.md): Guide for running experiments
- [`docs/training_tips.md`](docs/training_tips.md): Tips for training NSM models

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

### Ways to Contribute

1. **Code Contributions**: Bug fixes, feature implementations, performance improvements
2. **Documentation**: Improving existing docs, adding examples, tutorials
3. **Research**: New architectures, experimental ideas, benchmarking
4. **Testing**: Writing test cases, improving coverage
5. **Examples**: Jupyter notebooks, use case demonstrations

### Getting Started

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Beyond Transformer Team** - *Initial work* - [NeoSynaptic AI]

See also the list of [contributors](https://github.com/yourusername/beyond_transformer/contributors) who participated in this project.

## Acknowledgments

- Inspired by the foundational work on Transformers, State Space Models, and Neural Turing Machines
- Built upon the excellent research from the AI community
- Thanks to all contributors and supporters of this project

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers directly.
