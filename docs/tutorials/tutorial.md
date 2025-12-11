# PULSE Tutorial

This tutorial provides a hands-on introduction to PULSEs (PULSE), covering basic concepts, implementation details, and practical usage examples.

## Table of Contents

1. [Introduction to PULSEs](#introduction-to-neural-state-machines)
2. [Core Concepts](#core-concepts)
3. [Installation and Setup](#installation-and-setup)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Building Complete Models](#building-complete-models)
7. [Training and Evaluation](#training-and-evaluation)
8. [Best Practices](#best-practices)

## Introduction to PULSEs

PULSEs (PULSE) are a novel approach to sequence modeling that maintain and update explicit state vectors, combining the strengths of recurrent models with Transformers while addressing their limitations.

### Why PULSE?

Traditional Transformers face challenges with:
- **Quadratic complexity**: O(n²) attention computation for sequence length n
- **Limited interpretability**: Implicit attention patterns
- **Memory inefficiency**: Full sequence storage required

PULSE address these by:
- **Explicit state management**: O(n·s) complexity where s ≪ n
- **Interpretable states**: Learnable and trackable state evolution
- **Efficient computation**: Linear scaling with sequence length

## Core Concepts

### State Vectors

State vectors are the fundamental units of memory in PULSE:

```python
import torch
from pulse import StatePropagator

# Create state vectors
batch_size = 32
state_dim = 128
num_states = 16

# Initialize state vectors
state_vectors = torch.randn(batch_size, num_states, state_dim)
print(f"State vectors shape: {state_vectors.shape}")
```

### Gated Updates

State updates use gated mechanisms (similar to LSTM/GRU):

```python
# Previous state and new input
prev_state = torch.randn(batch_size, state_dim)
new_input = torch.randn(batch_size, state_dim)

# Create propagator with GRU-style gating
propagator = StatePropagator(state_dim=state_dim, gate_type='gru')

# Apply gated update
updated_state = propagator(prev_state, new_input)
print(f"Updated state shape: {updated_state.shape}")
```

### State-to-State Communication

States can communicate with each other using attention:

```python
# Multiple states with communication
multi_propagator = StatePropagator(
    state_dim=state_dim, 
    gate_type='gru',
    num_heads=4,
    enable_communication=True
)

# Process multiple states
prev_states = torch.randn(batch_size, num_states, state_dim)
new_inputs = torch.randn(batch_size, num_states, state_dim)
updated_states = multi_propagator(prev_states, new_inputs)
print(f"Multi-state update shape: {updated_states.shape}")
```

## Installation and Setup

### Prerequisites

Ensure you have Python 3.8+ and pip installed.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/beyond_transformer.git
cd beyond_transformer

# Create virtual environment (recommended)
python -m venv pulse_env
source pulse_env/bin/activate  # On Windows: pulse_env\Scripts\activate

# Install dependencies
pip install -r requirements/requirements.txt

# Install the package
pip install -e .
```

### Verification

```python
# Test installation
import torch
from pulse import StatePropagator

# Simple test
propagator = StatePropagator(state_dim=64)
x = torch.randn(2, 64)
y = propagator(x, x)
print("Installation successful!")
print(f"Output shape: {y.shape}")
```

## Basic Usage

### 1. State Propagation

Let's start with basic state propagation:

```python
import torch
from pulse import StatePropagator

# Create state propagator
state_dim = 128
propagator = StatePropagator(
    state_dim=state_dim,
    gate_type='gru',              # or 'lstm'
    enable_communication=False     # Disable for now
)

# Single state update
batch_size = 16
prev_state = torch.randn(batch_size, state_dim)
new_input = torch.randn(batch_size, state_dim)

updated_state = propagator(prev_state, new_input)
print(f"Previous state: {prev_state.shape}")
print(f"New input: {new_input.shape}")
print(f"Updated state: {updated_state.shape}")

# Verify state update
diff = torch.mean(torch.abs(updated_state - prev_state))
print(f"Average state change: {diff.item():.4f}")
```

### 2. Multi-State Processing

Process multiple states simultaneously:

```python
import torch
from pulse import StatePropagator

# Create multi-state propagator
state_dim = 128
num_states = 8
propagator = StatePropagator(
    state_dim=state_dim,
    gate_type='gru',
    num_heads=4,
    enable_communication=True     # Enable state-to-state communication
)

# Multi-state update with communication
batch_size = 16
prev_states = torch.randn(batch_size, num_states, state_dim)
new_inputs = torch.randn(batch_size, num_states, state_dim)

print(f"Previous states: {prev_states.shape}")
print(f"New inputs: {new_inputs.shape}")

# Apply update
updated_states = propagator(prev_states, new_inputs)
print(f"Updated states: {updated_states.shape}")

# Check communication effect
original_diff = torch.mean(torch.abs(updated_states - prev_states))
print(f"Average state change: {original_diff.item():.4f}")
```

### 3. Dynamic State Management

Manage states dynamically:

```python
import torch
from pulse import StateManager

# Create state manager with dynamic allocation
state_dim = 128
state_manager = StateManager(
    state_dim=state_dim,
    max_states=64,
    initial_states=16,
    prune_threshold=0.3
)

# Get current states
states = state_manager()
print(f"Initial states: {states.shape}")

# Simulate training process
for epoch in range(10):
    # Get importance scores
    scores = state_manager.get_importance_scores()
    active_count = state_manager.get_active_count()
    
    print(f"Epoch {epoch}: Active states = {active_count}, "
          f"Avg importance = {torch.mean(scores[:active_count]).item():.3f}")
    
    # Periodically manage states
    if epoch % 3 == 0:
        pruned = state_manager.prune_low_importance_states()
        allocated = state_manager.allocate_states(2)
        print(f"  Pruned {pruned} states, allocated {allocated} states")

# Final state count
final_states = state_manager()
print(f"Final states: {final_states.shape}")
```

## Advanced Features

### 1. Token-to-State Routing

Route input tokens to appropriate state nodes:

```python
import torch
from pulse import TokenToStateRouter

# Create token-to-state router
token_dim = 64
state_dim = 128
num_states = 8

router = TokenToStateRouter(
    token_dim=token_dim,
    state_dim=state_dim,
    num_states=num_states,
    num_heads=4
)

# Process sequence of tokens
batch_size = 16
seq_len = 20
tokens = torch.randn(batch_size, seq_len, token_dim)
states = torch.randn(batch_size, num_states, state_dim)

# Route tokens to states
routed_tokens, routing_weights = router(tokens, states)

print(f"Input tokens: {tokens.shape}")
print(f"State vectors: {states.shape}")
print(f"Routed tokens: {routed_tokens.shape}")
print(f"Routing weights: {routing_weights.shape}")

# Analyze routing
print(f"Average routing weight: {torch.mean(routing_weights).item():.3f}")
print(f"Max routing weight: {torch.max(routing_weights).item():.3f}")
print(f"Min routing weight: {torch.min(routing_weights).item():.3f}")
```

### 2. Complete PULSE Layer

Combine all components in a complete PULSE layer:

```python
import torch
from pulse import PulseLayer

# Create PULSE layer
state_dim = 128
token_dim = 64
num_states = 8

pulse_layer = PulseLayer(
    state_dim=state_dim,
    token_dim=token_dim,
    num_heads=4
)

# Process sequence through PULSE layer
batch_size = 16
seq_len = 20
states = torch.randn(batch_size, num_states, state_dim)
tokens = torch.randn(batch_size, seq_len, token_dim)

print(f"Initial states: {states.shape}")
print(f"Input tokens: {tokens.shape}")

# Process through PULSE layer
updated_states = pulse_layer(states, tokens)

print(f"Updated states: {updated_states.shape}")

# Show state evolution
state_diff = torch.mean(torch.abs(updated_states - states))
print(f"Average state change: {state_diff.item():.4f}")
```

### 3. State Importance Scoring

Track and analyze state importance:

```python
import torch
from pulse import StateManager
import matplotlib.pyplot as plt

# Create state manager
state_manager = StateManager(
    state_dim=64,
    max_states=32,
    initial_states=16,
    prune_threshold=0.2
)

# Simulate state importance evolution
importance_history = []

for step in range(50):
    # Get current importance scores
    scores = state_manager.get_importance_scores()
    active_count = state_manager.get_active_count()
    
    # Store average importance of active states
    avg_importance = torch.mean(scores[:active_count]).item()
    importance_history.append(avg_importance)
    
    # Simulate training updates that affect importance scores
    # (In practice, this would happen through backpropagation)
    
    # Periodic pruning
    if step % 10 == 0 and step > 0:
        pruned = state_manager.prune_low_importance_states()
        print(f"Step {step}: Pruned {pruned} states")

# Plot importance evolution
plt.figure(figsize=(10, 6))
plt.plot(importance_history)
plt.xlabel('Training Steps')
plt.ylabel('Average State Importance')
plt.title('State Importance Evolution')
plt.grid(True, alpha=0.3)
plt.show()

print("State importance evolution plotted!")
```

## Building Complete Models

### 1. Simple PULSE Classifier

Build a complete classifier using PULSE components:

```python
import torch
import torch.nn as nn
from pulse.models import SimplePulse

# Create PULSE classifier for MNIST-like data
model = SimplePulse(
    input_dim=784,      # Flattened 28x28 images
    state_dim=128,
    num_states=16,
    output_dim=10,      # 10 digit classes
    gate_type='gru'
)

# Test forward pass
batch_size = 32
x = torch.randn(batch_size, 784)
output = model(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

# Test backward pass
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy labels
labels = torch.randint(0, 10, (batch_size,))

# Training step
loss = criterion(output, labels)
loss.backward()
optimizer.step()

print(f"Training step completed. Loss: {loss.item():.4f}")
```

### 2. Sequence Processing Model

Build a model for sequence processing:

```python
import torch
import torch.nn as nn
from pulse import PulseLayer, StateManager

class SequencePulse(nn.Module):
    def __init__(self, token_dim=64, state_dim=128, num_states=16, output_dim=10):
        super().__init__()
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.output_dim = output_dim
        
        # State manager for dynamic state allocation
        self.state_manager = StateManager(
            state_dim=state_dim,
            max_states=num_states * 2,
            initial_states=num_states,
            prune_threshold=0.3
        )
        
        # PULSE layers
        self.pulse_layer1 = PulseLayer(state_dim, token_dim, num_heads=4)
        self.pulse_layer2 = PulseLayer(state_dim, state_dim, num_heads=4)
        
        # Output projection
        self.output_projection = nn.Linear(state_dim * num_states, output_dim)
        
    def forward(self, tokens):
        # Get initial states
        states = self.state_manager()
        batch_size = tokens.size(0)
        
        # Expand states to batch dimension
        states = states.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through PULSE layers
        states = self.pulse_layer1(states, tokens)
        states = self.pulse_layer2(states, states)  # Self-processing
        
        # Global pooling and output projection
        pooled_states = states.view(batch_size, -1)
        output = self.output_projection(pooled_states)
        
        return output

# Create and test model
model = SequencePulse(token_dim=64, state_dim=128, num_states=16, output_dim=5)

# Test with sequence data
batch_size = 8
seq_len = 30
tokens = torch.randn(batch_size, seq_len, 64)

output = model(tokens)
print(f"Input tokens: {tokens.shape}")
print(f"Output: {output.shape}")

# Verify differentiability
loss = output.sum()
loss.backward()
print("Backward pass successful!")
```

## Training and Evaluation

### 1. Basic Training Loop

Implement a basic training loop:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=64, n_classes=5, 
                          n_informative=32, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model
from pulse.models import SimplePulse
model = SimplePulse(input_dim=64, state_dim=64, num_states=8, output_dim=5)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 20
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    scheduler.step()
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print()

print("Training completed!")
```

### 2. Advanced Training with Logging

Implement advanced training with comprehensive logging:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
import os

class TrainingLogger:
    def __init__(self, log_dir="runs", experiment_name="pulse_training"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = {}
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def log_metrics(self, epoch, train_metrics, val_metrics):
        # Log training metrics
        for key, value in train_metrics.items():
            self.log_scalar(f"train/{key}", value, epoch)
            
        # Log validation metrics
        for key, value in val_metrics.items():
            self.log_scalar(f"val/{key}", value, epoch)
            
        # Store metrics
        self.metrics[epoch] = {
            'train': train_metrics,
            'val': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
    def save_metrics(self):
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def close(self):
        self.writer.close()
        self.save_metrics()

# Advanced training example
def advanced_train(model, train_loader, val_loader, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger = TrainingLogger(experiment_name="pulse_advanced_training")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_total += target.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        train_metrics = {
            'loss': avg_train_loss,
            'accuracy': train_acc
        }
        
        val_metrics = {
            'loss': avg_val_loss,
            'accuracy': val_acc
        }
        
        # Log metrics
        logger.log_metrics(epoch, train_metrics, val_metrics)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 
                      os.path.join(logger.log_dir, "best_model.pth"))
        
        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch {epoch}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
            print(f'  LR: {scheduler.get_last_lr()[0]:.6f}')
            print()
    
    logger.close()
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Logs saved to: {logger.log_dir}")

# Example usage would go here
```

## Best Practices

### 1. Model Design

```python
# Good practices for PULSE model design

# 1. Choose appropriate state dimensions
# Rule of thumb: state_dim ≈ 2-4 × typical token_dim
state_dim = 128  # For token_dim=64

# 2. Scale number of states appropriately
# Start small and increase based on task complexity
num_states = 8   # Start with 8-16 states

# 3. Use appropriate gating mechanisms
# GRU for simpler tasks, LSTM for complex memory dependencies
gate_type = 'gru'  # Usually sufficient for most tasks

# 4. Enable communication for relational tasks
enable_communication = True  # For tasks requiring state interaction

# 5. Use dynamic state management for variable complexity tasks
use_dynamic_states = True
prune_threshold = 0.3  # Prune states with low importance scores
```

### 2. Training Strategies

```python
# Effective training strategies

# 1. Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 2. Gradient clipping for stability
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Mixed precision training for efficiency
scaler = torch.cuda.amp.GradScaler()
# In training loop:
# with torch.cuda.amp.autocast():
#     output = model(data)
#     loss = criterion(output, target)
# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()

# 4. Early stopping
# Monitor validation metrics and stop when no improvement

# 5. Regularization
# Add dropout, weight decay, and batch normalization as needed
```

### 3. Memory Management

```python
# Efficient memory usage strategies

# 1. Use appropriate batch sizes
# Start with smaller batches and increase as needed
batch_size = 32  # Start small, increase if memory allows

# 2. Clear cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 3. Use gradient checkpointing for deep models
# torch.utils.checkpoint for memory-intensive computations

# 4. Process sequences in chunks for very long sequences
def chunked_processing(model, long_sequence, chunk_size=1000):
    """Process long sequences in chunks to manage memory."""
    results = []
    for i in range(0, len(long_sequence), chunk_size):
        chunk = long_sequence[i:i+chunk_size]
        result = model(chunk)
        results.append(result)
    return torch.cat(results, dim=0)

# 5. Monitor memory usage
def monitor_memory():
    """Monitor GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

### 4. Debugging and Monitoring

```python
# Debugging and monitoring tools

# 1. Use the built-in debugger
from pulse.utils.debugger import PulseDebugger
debugger = PulseDebugger(log_dir="debug_logs", verbose=True)
debugger.enable_debug()

# 2. Log important metrics during training
def log_training_step(step, loss, accuracy, lr):
    """Log training metrics."""
    print(f"Step {step}: Loss={loss:.4f}, Acc={accuracy:.2f}%, LR={lr:.6f}")

# 3. Monitor gradients
def check_gradients(model, threshold=10.0):
    """Check for exploding gradients."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > threshold:
        print(f"Warning: Gradient norm is high: {total_norm:.2f}")
    
    return total_norm

# 4. Visualization tools
from pulse.visualization import PULSEVisualizer
visualizer = PULSEVisualizer()

# Plot attention weights
# visualizer.plot_attention_map(attention_weights)

# Plot state evolution
# visualizer.plot_state_evolution(state_trajectories)
```

This tutorial provides a comprehensive introduction to PULSEs, covering everything from basic concepts to advanced usage patterns. By following these examples and best practices, you'll be well-equipped to build and train your own PULSE models for a variety of sequence processing tasks.