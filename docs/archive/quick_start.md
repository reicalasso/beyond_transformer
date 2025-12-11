# PULSEs: Quick Start Guide
## Get Running with PULSE in 5 Minutes

---

## 🚀 **Installation & Setup**

### **Prerequisites**
- Python 3.8+ (3.10 recommended)
- PyTorch 1.9+ (2.0+ recommended for optimal performance)
- CUDA (optional, but recommended for GPU acceleration)

### **Option 1: pip Installation (Recommended)**

```bash
# Install from PyPI (when available)
pip install beyond-transformer

# Or install from source
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
pip install -e ".[all]"
```

### **Option 2: Conda Environment**

```bash
# Create optimized environment
git clone https://github.com/reicalasso/beyond_transformer.git
cd beyond_transformer
conda env create -f environment.yml
conda activate beyond_transformer
```

### **Verify Installation**

```python
import torch
from pulse import PulseLayer, StateManager
print("✅ PULSEs ready!")
```

---

## 📚 **Basic Usage Examples**

### **Example 1: Simple Classification (30 seconds)**

```python
import torch
from pulse.models import SimplePulse

# Create a classification model
model = SimplePulse(
    input_dim=768,      # Input feature dimension
    state_dim=256,      # Memory state dimension
    num_states=16,      # Number of memory states
    output_dim=10,      # Number of classes
    gate_type='gru'     # Gating mechanism (gru/lstm)
)

# Sample data (batch_size=32, sequence_length=512, features=768)
batch_data = torch.randn(32, 512, 768)

# Forward pass
predictions = model(batch_data)
print(f"Predictions shape: {predictions.shape}")  # [32, 10]

print(f"State updated! Shape: {updated_state.shape}")
```

### 3. Complete Model

```python
from pulse.models import SimplePulse

# Create PULSE model
model = SimplePulse(
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
from pulse import PulseLayer

# Process sequences with PULSE
pulse_layer = PulseLayer(
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
    states = pulse_layer(states, token_t.unsqueeze(1).repeat(1, 16, 1))
```

### 2. Dynamic State Management

```python
from pulse import StateManager

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
3. **Run Experiments**: Try scripts in `src/pulse/experiments/`
4. **Read Documentation**: Browse detailed docs in `docs/`

## 🤝 Need Help?

- Check the [FAQ](faq.md) for common questions
- Open an [issue](https://github.com/yourusername/beyond_transformer/issues) for bugs or feature requests
- Join our [discussions](https://github.com/yourusername/beyond_transformer/discussions) for community support

Ready to explore PULSEs? [Start experimenting!](../notebooks/getting_started.ipynb)