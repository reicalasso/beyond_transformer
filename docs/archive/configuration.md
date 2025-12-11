# Configuration Management

This document describes how configuration is managed in the Beyond Transformer project.

## Overview

The project uses a flexible configuration system that supports both JSON and YAML formats. Configuration files are used to define model parameters, training settings, data specifications, and experiment details.

## Directory Structure

```
configs/
├── default_config.json      # Default configuration
├── small_model_config.json  # Configuration for small models
├── large_model_config.json  # Configuration for large models
├── long_training_config.json # Configuration for long training experiments
└── debug_config.yaml        # Debug configuration in YAML format
```

## Configuration File Format

Configuration files can be in either JSON or YAML format. Both formats support the same structure and features.

### JSON Example

```json
{
  "model": {
    "input_dim": 784,
    "state_dim": 128,
    "num_states": 16,
    "output_dim": 10,
    "gate_type": "gru"
  },
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "data": {
    "dataset": "mnist",
    "num_samples": 1000
  }
}
```

### YAML Example

```yaml
model:
  input_dim: 784
  state_dim: 128
  num_states: 16
  output_dim: 10
  gate_type: "gru"

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"

data:
  dataset: "mnist"
  num_samples: 1000
```

## Configuration Management API

The project includes a `Config` class in `src/pulse/config.py` for managing configuration files.

### Loading Configuration

```python
from pulse.config import Config

# Load configuration from file
config = Config("configs/default_config.json")

# Access configuration values
input_dim = config.get('model.input_dim')
epochs = config['training.epochs']
```

### Setting Configuration Values

```python
# Set configuration values
config['model.dropout'] = 0.1
config.set('training.epochs', 20)
```

### Saving Configuration

```python
# Save configuration to file
config.save("configs/modified_config.json")
```

### Validating Configuration

```python
# Validate that required keys are present
required_keys = ['model.input_dim', 'training.epochs', 'model.state_dim']
if config.validate(required_keys):
    print("Configuration is valid")
else:
    print("Configuration is missing required keys")
```

## Using Configuration in Scripts

Configuration files are used in training scripts to define model and training parameters.

### Example Training Script

```python
import argparse
from pulse.config import Config
from pulse.models import SimplePulse

def main():
    parser = argparse.ArgumentParser(description='Train PULSE model')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create model from configuration
    model = SimplePulse(
        input_dim=config['model.input_dim'],
        state_dim=config['model.state_dim'],
        num_states=config['model.num_states'],
        output_dim=config['model.output_dim'],
        gate_type=config['model.gate_type']
    )
    
    # Training logic using config['training.epochs'], etc.
    # ...

if __name__ == "__main__":
    main()
```

## Creating New Configuration Files

To create a new configuration file:

1. Copy an existing configuration file as a template
2. Modify the parameters as needed
3. Save with a descriptive name in the `configs/` directory
4. Update documentation if necessary

## Best Practices

- **Use descriptive names** for configuration files
- **Document non-obvious parameters** in comments or separate documentation
- **Validate configurations** before use in critical applications
- **Version control configurations** for reproducibility
- **Separate environment-specific settings** (e.g., paths) from core parameters
- **Use consistent naming conventions** for keys across different configuration files