# Beyond Transformer Documentation

Welcome to the Beyond Transformer documentation. This project explores alternatives to the Transformer architecture, specifically focusing on Neural State Machines (NSM).

## Table of Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Experiments](#experiments)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## Installation

To install the package, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer
pip install -e .
```

Or using conda:

```bash
conda env update -f environment.yml
conda activate beyond_transformer
```

## Getting Started

### Training a Model

To train a model, use the provided training script:

```bash
python scripts/train_model.py --config configs/default_config.json
```

### Running Experiments

To run experiments, check the `notebooks/` directory for Jupyter notebooks or use the experiment scripts in `src/nsm/experiments/`.

## Architecture

The core of this project is the Neural State Machine (NSM) architecture. Key components include:

- **StatePropagator**: Implements gated updates for state vectors.
- **TokenToStateRouter**: Routes input tokens to appropriate state nodes.
- **StateManager**: Manages state nodes with dynamic allocation and pruning.
- **NSMLayer**: Combines state updates with token-to-state routing.
- **HybridAttention**: Combines token-to-state routing with content-based attention.

## Experiments

The project includes several experiments to evaluate the NSM architecture:

- **State Count Sweep**: Tests different numbers of state nodes.
- **Dynamic State Allocation**: Evaluates dynamic state allocation and pruning.

## API Reference

For detailed API documentation, see the [API Reference](api_reference.md).

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.