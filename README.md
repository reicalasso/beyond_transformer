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

### 3. Dynamic State Allocation and Pruning
- Learnable importance scores for each state node
- Automatic pruning of low-importance states
- Dynamic allocation of new states when needed

### 4. Hybrid Attention Mechanisms
- Token-to-state routing with learned attention
- Content-based attention for information flow
- Flexible routing strategies

## Project Structure

```
.
├── .github/
│   └── workflows/              # CI/CD workflows
├── configs/                    # Configuration files
├── data/                       # Data directory (not committed)
├── docs/                       # Documentation
├── notebooks/                  # Jupyter notebooks
├── references/                 # Reference materials
├── requirements/               # Dependency files
│   ├── requirements.txt
│   ├── requirements-experiments.txt
│   └── requirements-test.txt
├── results/                    # Experiment results
├── scripts/                    # Utility scripts
├── src/                        # Source code
│   └── nsm/                    # Neural State Machine implementation
└── tests/                      # Test files
```

For more detailed structure of `src/nsm/`, see [src/README.md](src/nsm/README.md).

## Installation

### Using pip

```bash
# Clone the repository
git clone <repository-url>
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
```

## Usage

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

## Experiments

### State Count Hyperparameter Sweep

Run the experiment to test different numbers of state nodes:

```bash
python scripts/experiments/run_experiments.py
```

Results are saved to `state_count_sweep_results.json` with metrics:
- Training accuracy
- Test accuracy
- Memory usage
- Training time

### Routing Visualization

Visualize token-to-state routing patterns with heatmaps and importance scores:

```bash
# Open the notebook
jupyter notebook notebooks/experiments/routing_viz.ipynb
```

The notebook provides several visualization types:
- Token-to-state routing heatmaps
- State importance score overlays
- Routing entropy analysis
- Multi-head routing comparison

### Dynamic State Allocation Experiment

Test dynamic state allocation and pruning mechanisms:

```bash
python src/nsm/experiments/dynamic_state_allocation.py
```

### Baseline Model Comparison

Compare NSM against traditional architectures (LSTM, GRU, Transformer):

```bash
# Run comparison experiment
python notebooks/scripts/run_baseline_comparison.py

# View results summary
python scripts/experiments/summarize_results.py
```

Results are saved to `results/experiments/baseline_comparison_results.json` with metrics:
- Accuracy (training and test)
- F1 Score (conceptual)
- Memory usage
- FLOPs (conceptual)
- Training speed

Datasets tested:
- MNIST (image classification)
- CIFAR-10 (image classification)
- Tiny Shakespeare (text generation) - partial results
- IMDb (sentiment classification) - partial results

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

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.