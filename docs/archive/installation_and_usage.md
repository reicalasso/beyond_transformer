# Installation and Usage Guide

This guide provides detailed instructions for installing and using the Beyond Transformer (Neural State Machine) project.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 8 GB RAM
- 2 GB available disk space

### Recommended Requirements
- Python 3.9 or higher
- 16 GB RAM or more
- CUDA-compatible GPU (for training)
- 10 GB available disk space

## Installation Options

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer

# Create virtual environment (recommended)
python -m venv beyond_transformer_env
source beyond_transformer_env/bin/activate  # On Windows: beyond_transformer_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements/requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer

# Create conda environment
conda env update -f environment.yml
conda activate beyond_transformer

# Install the package
pip install -e .
```

### Option 3: Development Installation

For contributing or running experiments:

```bash
# Install development dependencies
pip install -r requirements/requirements-experiments.txt
pip install -r requirements/requirements-test.txt

# Install in development mode
pip install -e .

# Optional: Install Jupyter for notebooks
pip install jupyterlab
```

## Verifying Installation

After installation, verify that everything works correctly:

```bash
# Test importing the main modules
python -c "import nsm; print('NSM imported successfully')"

# Run basic tests
python -m pytest tests/ -v --tb=short

# Test core functionality
python -c "
import torch
from nsm import StatePropagator
prop = StatePropagator(state_dim=64)
x = torch.randn(2, 64)
y = prop(x, x)
print('StatePropagator test passed')
"
```

## Basic Usage Examples

### 1. Simple State Propagation

```python
import torch
from nsm import StatePropagator

# Create state propagator
propagator = StatePropagator(
    state_dim=128,
    gate_type='gru',              # or 'lstm'
    enable_communication=True      # Enable state-to-state communication
)

# Single state update
batch_size = 32
prev_state = torch.randn(batch_size, 128)
new_input = torch.randn(batch_size, 128)
updated_state = propagator(prev_state, new_input)

print(f"Previous state shape: {prev_state.shape}")
print(f"Updated state shape: {updated_state.shape}")
```

### 2. Multi-State Propagation

```python
import torch
from nsm import StatePropagator

# Create state propagator for multiple states
propagator = StatePropagator(
    state_dim=128,
    gate_type='gru',
    enable_communication=True
)

# Multiple state update with communication
batch_size = 32
num_states = 16
prev_states = torch.randn(batch_size, num_states, 128)
new_inputs = torch.randn(batch_size, num_states, 128)
updated_states = propagator(prev_states, new_inputs)

print(f"Previous states shape: {prev_states.shape}")
print(f"Updated states shape: {updated_states.shape}")
```

### 3. Complete NSM Model

```python
import torch
from nsm.models import SimpleNSM

# Create a complete NSM model
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

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# Training setup
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Simple training step
labels = torch.randint(0, 10, (batch_size,))  # Random labels
loss = criterion(output, labels)
loss.backward()
optimizer.step()

print(f"Training step completed. Loss: {loss.item():.4f}")
```

### 4. Dynamic State Management

```python
import torch
from nsm import StateManager, NSMLayer

# Create state manager with dynamic allocation
state_manager = StateManager(
    state_dim=128,
    max_states=64,
    initial_states=16,
    prune_threshold=0.3
)

# Create NSM layer
nsm_layer = NSMLayer(
    state_dim=128,
    token_dim=64,
    num_heads=4
)

# Get current active states
states = state_manager()
print(f"Initial active states: {states.shape}")

# Simulate training process with dynamic state management
for epoch in range(5):
    # Periodic state management
    if epoch % 2 == 0:
        pruned = state_manager.prune_low_importance_states()
        allocated = state_manager.allocate_states(2)
        print(f"Epoch {epoch}: Pruned {pruned}, Allocated {allocated}")
        print(f"Active states: {state_manager.get_active_count()}")

# Final state count
print(f"Final active states: {state_manager.get_active_count()}")
```

## Advanced Usage Examples

### 1. Custom Configuration

```python
import json
from nsm.config import Config

# Load configuration
config = Config("configs/default_config.json")

# Access configuration values
state_dim = config.get('model.state_dim', 128)
num_states = config.get('model.num_states', 16)
gate_type = config.get('model.gate_type', 'gru')

print(f"Loaded configuration:")
print(f"  State dimension: {state_dim}")
print(f"  Number of states: {num_states}")
print(f"  Gate type: {gate_type}")

# Modify configuration
config.set('training.learning_rate', 0.0001)
config.set('training.epochs', 100)

# Save modified configuration
config.save("configs/custom_config.json")
```

### 2. Experiment Tracking

```python
import torch
from nsm.experiment import ExperimentRunner

# Create experiment runner
runner = ExperimentRunner(
    config_file="configs/default_config.json",
    experiment_name="state_count_sweep"
)

# Run experiment with different state counts
state_counts = [8, 16, 32, 64]
for state_count in state_counts:
    # Update configuration
    runner.config.set('model.num_states', state_count)
    
    # Run experiment
    results = runner.run_experiment()
    
    print(f"State count {state_count}:")
    print(f"  Training accuracy: {results['train_accuracy']:.2f}%")
    print(f"  Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"  Memory usage: {results['memory_usage']:.2f} MB")
    print(f"  Training time: {results['training_time']:.2f} seconds")
```

### 3. Visualization and Analysis

```python
import torch
from nsm.visualization import NSMVisualizer

# Create visualizer
visualizer = NSMVisualizer()

# Generate sample data for visualization
attention_weights = torch.softmax(torch.randn(8, 8), dim=-1)
memory_content = torch.randn(16, 20)
state_trajectories = [torch.randn(8, 16) for _ in range(5)]

# Plot attention map
visualizer.plot_attention_map(
    attention_weights,
    title="Sample Attention Map",
    x_labels=[f"Pos{i}" for i in range(8)],
    y_labels=[f"Query{i}" for i in range(8)]
)

# Plot memory content
visualizer.plot_memory_content(
    memory_content,
    title="Sample Memory Content"
)

# Plot state evolution
visualizer.plot_state_evolution(
    state_trajectories,
    title="State Evolution Over Time"
)

# Save visualization report
report_dir = visualizer.create_comprehensive_report({
    'attention_weights': attention_weights,
    'memory_content': memory_content,
    'state_trajectories': state_trajectories
})

print(f"Visualization report saved to: {report_dir}")
```

## Running Experiments

### Using Configuration Files

The project supports both JSON and YAML configuration files:

```bash
# Run with JSON configuration
python scripts/train_model.py --config configs/default_config.json

# Run with YAML configuration
python scripts/train_model.py --config configs/debug_config.yaml

# Run with custom configuration
python scripts/train_model.py --config configs/custom_experiment.json
```

### Experiment Scripts

Several pre-built experiment scripts are available:

```bash
# Run state count hyperparameter sweep
python src/nsm/experiments/state_count_sweep.py

# Run dynamic state allocation experiment
python src/nsm/experiments/dynamic_state_allocation.py

# Run baseline comparison
python notebooks/experiments/baseline_comparison.py

# Run hyperparameter sweep
python notebooks/experiments/hyperparameter_sweep.py
```

### Monitoring and Logging

```python
import logging
from nsm.utils.logger import NSMLogger

# Setup logging
logger = NSMLogger(
    log_dir="experiment_logs",
    experiment_name="training_experiment"
)

# Log training metrics
logger.log_metrics({
    'loss': 0.456,
    'accuracy': 0.876,
    'learning_rate': 0.001
}, step=100)

# Log memory content
memory = torch.randn(32, 128)
logger.log_memory_content(memory, step=100, batch_idx=0)

# Log attention weights
attention = torch.softmax(torch.randn(32, 4, 8, 8), dim=-1)
logger.log_attention_weights(attention, step=100, batch_idx=0)

# Log state variables
states = torch.randn(32, 16, 128)
logger.log_state_variables(states, step=100, batch_idx=0, state_type="hidden")
```

## Troubleshooting Common Issues

### 1. Import Errors

If you encounter import errors:

```bash
# Make sure you're in the project root directory
cd /path/to/beyond_transformer

# Install in development mode
pip install -e .

# Or add the src directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/beyond_transformer/src"
```

### 2. CUDA Issues

For CUDA-related problems:

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is not available, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Memory Issues

For memory-related problems:

```bash
# Reduce batch size in configuration
# In your config file:
{
    "training": {
        "batch_size": 16,  # Reduce from default
        "gradient_accumulation_steps": 2  # Increase to maintain effective batch size
    }
}

# Use mixed precision training
python scripts/train_model.py --config configs/default_config.json --mixed_precision
```

### 4. Performance Issues

For performance optimization:

```bash
# Use multiple workers for data loading
# In your config file:
{
    "training": {
        "num_workers": 4,
        "pin_memory": true
    }
}

# Enable AMP (Automatic Mixed Precision)
python scripts/train_model.py --amp --config configs/default_config.json
```

## Best Practices

### 1. Model Development

```python
# Always start with small models for testing
# configs/debug_config.json
{
    "model": {
        "state_dim": 64,
        "num_states": 8,
        "token_dim": 32
    },
    "training": {
        "batch_size": 8,
        "epochs": 2
    }
}

# Test with debug configuration first
python scripts/train_model.py --config configs/debug_config.json
```

### 2. Experiment Tracking

```python
# Use descriptive experiment names
EXPERIMENT_NAME = f"state_count_sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"

# Log all important parameters
experiment_params = {
    'state_dim': 128,
    'num_states': 16,
    'gate_type': 'gru',
    'learning_rate': 0.001,
    'batch_size': 32
}

# Save experiment configuration
with open(f"results/{EXPERIMENT_NAME}/config.json", 'w') as f:
    json.dump(experiment_params, f, indent=2)
```

### 3. Resource Management

```python
# Clean up GPU memory
import torch
torch.cuda.empty_cache()

# Monitor memory usage
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Next Steps

1. **Explore Examples**: Check the `notebooks/` directory for Jupyter notebooks with detailed examples
2. **Run Experiments**: Try the experiment scripts in `src/nsm/experiments/`
3. **Read Documentation**: Review the detailed documentation in `docs/`
4. **Contribute**: Follow our [Contributing Guidelines](CONTRIBUTING.md) to help improve the project

Happy coding with Neural State Machines!