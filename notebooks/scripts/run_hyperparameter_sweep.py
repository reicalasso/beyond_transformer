#!/usr/bin/env python3
"""
State Count Hyperparameter Sweep

This script implements a hyperparameter sweep to test different numbers of state nodes 
(8, 16, 32, 64) across multiple datasets (MNIST, Tiny Shakespeare, IMDb). We'll record 
accuracy, memory usage, and training speed, then plot performance vs state count for 
each dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import os
import sys
import json
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nsm.modules.state_propagator import StatePropagator

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SimpleNSMModel(nn.Module):
    """A simple model using StatePropagator for hyperparameter testing."""
    
    def __init__(self, input_dim, state_dim, num_states, output_dim, gate_type='gru'):
        super(SimpleNSMModel, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.output_dim = output_dim
        
        # Input embedding to state dimension
        if input_dim > 1000:  # Discrete data (text)
            self.input_embedding = nn.Embedding(input_dim, state_dim)
        else:  # Continuous data (images)
            self.input_embedding = nn.Linear(input_dim, state_dim)
        
        # State propagator
        self.state_propagator = StatePropagator(
            state_dim=state_dim, 
            gate_type=gate_type,
            enable_communication=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(state_dim * num_states, output_dim)
        
        # Initialize states
        self.initial_states = nn.Parameter(torch.randn(1, num_states, state_dim))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len] for discrete data or [batch_size, input_dim] for continuous
        batch_size = x.size(0)
        
        # Handle different input types
        if x.dim() == 2 and x.size(1) > 1000:  # Discrete sequences (text)
            # Embed input to state dimension
            x = self.input_embedding(x)  # [batch_size, seq_len, state_dim]
            seq_len = x.size(1)
        elif x.dim() == 2:  # Continuous features (images)
            # Project input to state dimension
            x = self.input_embedding(x)  # [batch_size, state_dim]
            x = x.unsqueeze(1)  # [batch_size, 1, state_dim]
            seq_len = 1
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Initialize states for batch
        states = self.initial_states.repeat(batch_size, 1, 1)  # [batch_size, num_states, state_dim]
        
        # Process sequence
        for t in range(seq_len):
            # For simplicity, we'll use the same input for all states
            if seq_len > 1:
                input_t = x[:, t, :].unsqueeze(1).repeat(1, self.num_states, 1)  # [batch_size, num_states, state_dim]
            else:
                input_t = x.repeat(1, self.num_states, 1)  # [batch_size, num_states, state_dim]
            
            # Update states
            states = self.state_propagator(states, input_t)
        
        # Global pooling of states
        pooled_states = states.view(batch_size, -1)  # [batch_size, state_dim * num_states]
        
        # Output projection
        output = self.output_projection(pooled_states)
        
        return output


def create_mnist_like_data(num_samples=1000):
    """Create MNIST-like synthetic data (28x28 grayscale images)."""
    # MNIST: 28x28 = 784 pixels, 10 classes
    X = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    return torch.utils.data.TensorDataset(X, y)


def create_tiny_shakespeare_like_data(num_samples=1000, seq_len=256):
    """Create Tiny Shakespeare-like synthetic data (sequences of characters)."""
    # Vocabulary size for characters (simplified)
    vocab_size = 100
    X = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    # For language modeling, target is next character
    y = torch.randint(0, vocab_size, (num_samples,), dtype=torch.long)
    return torch.utils.data.TensorDataset(X, y)


def create_imdb_like_data(num_samples=1000, seq_len=512):
    """Create IMDb-like synthetic data (sequences of words)."""
    # Vocabulary size for words (simplified)
    vocab_size = 10000
    X = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    # Binary sentiment classification
    y = torch.randint(0, 2, (num_samples,))
    return torch.utils.data.TensorDataset(X, y)


def train_model(model, train_loader, epochs=3, lr=0.001):
    """Train the model and return metrics."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Memory tracking
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Training time tracking
    start_time = time.time()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
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
            
            # Limit training for demonstration
            if batch_idx > 10:  # Just for quick testing
                break
    
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    training_time = end_time - start_time
    memory_usage = final_memory - initial_memory
    accuracy = 100. * correct / total
    
    return {
        'accuracy': accuracy,
        'memory_usage': memory_usage,
        'training_time': training_time
    }


def evaluate_model(model, test_loader):
    """Evaluate the model and return accuracy."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Limit evaluation for demonstration
            if total > 100:  # Just for quick testing
                break
    
    accuracy = 100. * correct / total
    return accuracy


def run_hyperparameter_sweep():
    """Run hyperparameter sweep for different state counts."""
    
    # State counts to test
    state_counts = [8, 16, 32, 64]
    
    # Results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # Dataset configurations
    datasets = {
        'MNIST': {
            'data_func': create_mnist_like_data,
            'input_dim': 784,
            'output_dim': 10,
        },
        'Tiny_Shakespeare': {
            'data_func': create_tiny_shakespeare_like_data,
            'input_dim': 100,  # vocab_size
            'output_dim': 100,  # vocab_size
        },
        'IMDb': {
            'data_func': create_imdb_like_data,
            'input_dim': 10000,  # vocab_size
            'output_dim': 2,     # binary classification
        }
    }
    
    # Fixed parameters
    state_dim = 128
    epochs = 3
    batch_size = 32
    
    for dataset_name, dataset_config in datasets.items():
        print(f"\n=== Testing {dataset_name} ===")
        
        # Create dataset
        train_dataset = dataset_config['data_func'](num_samples=500)  # Smaller for demo
        test_dataset = dataset_config['data_func'](num_samples=100)   # Smaller for demo
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        for num_states in state_counts:
            print(f"  Testing with {num_states} states...")
            
            # Create model
            model = SimpleNSMModel(
                input_dim=dataset_config['input_dim'],
                state_dim=state_dim,
                num_states=num_states,
                output_dim=dataset_config['output_dim']
            )
            
            # Train model
            metrics = train_model(model, train_loader, epochs=epochs)
            
            # Evaluate model
            test_accuracy = evaluate_model(model, test_loader)
            
            # Store results
            results[dataset_name]['state_counts'].append(num_states)
            results[dataset_name]['accuracies'].append(metrics['accuracy'])
            results[dataset_name]['test_accuracies'].append(test_accuracy)
            results[dataset_name]['memory_usages'].append(metrics['memory_usage'])
            results[dataset_name]['training_times'].append(metrics['training_time'])
            
            print(f"    Train Accuracy: {metrics['accuracy']:.2f}%")
            print(f"    Test Accuracy: {test_accuracy:.2f}%")
            print(f"    Memory Usage: {metrics['memory_usage']:.2f} MB")
            print(f"    Training Time: {metrics['training_time']:.2f} seconds")
    
    return results


def plot_results(results):
    """Plot the results of the hyperparameter sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('State Count Hyperparameter Sweep Results', fontsize=16)
    
    # Plot 1: Training Accuracy vs State Count
    ax = axes[0, 0]
    for dataset_name, data in results.items():
        ax.plot(data['state_counts'], data['accuracies'], marker='o', label=dataset_name)
    ax.set_xlabel('Number of State Nodes')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy vs State Count')
    ax.legend()
    ax.grid(True)
    
    # Plot 2: Test Accuracy vs State Count
    ax = axes[0, 1]
    for dataset_name, data in results.items():
        ax.plot(data['state_counts'], data['test_accuracies'], marker='o', label=dataset_name)
    ax.set_xlabel('Number of State Nodes')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs State Count')
    ax.legend()
    ax.grid(True)
    
    # Plot 3: Memory Usage vs State Count
    ax = axes[1, 0]
    for dataset_name, data in results.items():
        ax.plot(data['state_counts'], data['memory_usages'], marker='o', label=dataset_name)
    ax.set_xlabel('Number of State Nodes')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage vs State Count')
    ax.legend()
    ax.grid(True)
    
    # Plot 4: Training Time vs State Count
    ax = axes[1, 1]
    for dataset_name, data in results.items():
        ax.plot(data['state_counts'], data['training_times'], marker='o', label=dataset_name)
    ax.set_xlabel('Number of State Nodes')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time vs State Count')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_sweep_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Run the hyperparameter sweep
    print("Running hyperparameter sweep...")
    results = run_hyperparameter_sweep()
    
    # Convert results to JSON-serializable format
    serializable_results = {}
    for dataset_name, data in results.items():
        serializable_results[dataset_name] = {
            'state_counts': [int(x) for x in data['state_counts']],
            'accuracies': [float(x) for x in data['accuracies']],
            'test_accuracies': [float(x) for x in data['test_accuracies']],
            'memory_usages': [float(x) for x in data['memory_usages']],
            'training_times': [float(x) for x in data['training_times']]
        }
    
    # Save results
    with open('results/experiments/hyperparameter_sweep_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Plot results
    plot_results(results)
    
    print("Results saved to results/experiments/hyperparameter_sweep_results.json")
    print("Plot saved to results/visualization/nsm_hyperparameter_sweep_results.png")
    print("Hyperparameter sweep completed successfully!")