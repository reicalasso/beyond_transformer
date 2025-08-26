#!/usr/bin/env python3
"""
Simple State Count Hyperparameter Sweep for MNIST

This script implements a hyperparameter sweep to test different numbers of state nodes 
(8, 16, 32, 64) on MNIST. We'll record accuracy, memory usage, and training speed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import psutil
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nsm.state_propagator import StatePropagator

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
        
        # Input projection to state dimension
        self.input_projection = nn.Linear(input_dim, state_dim)
        
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
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # Project input to state dimension
        x = self.input_projection(x)  # [batch_size, state_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, state_dim]
        
        # Initialize states for batch
        states = self.initial_states.repeat(batch_size, 1, 1)  # [batch_size, num_states, state_dim]
        
        # Update states (single step for MNIST)
        input_t = x.repeat(1, self.num_states, 1)  # [batch_size, num_states, state_dim]
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
    results = {
        'state_counts': [],
        'accuracies': [],
        'test_accuracies': [],
        'memory_usages': [],
        'training_times': []
    }
    
    # Dataset configuration
    input_dim = 784
    output_dim = 10
    
    # Fixed parameters
    state_dim = 128
    epochs = 3
    batch_size = 32
    
    print("=== Testing MNIST ===")
    
    # Create dataset
    train_dataset = create_mnist_like_data(num_samples=500)  # Smaller for demo
    test_dataset = create_mnist_like_data(num_samples=100)   # Smaller for demo
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    for num_states in state_counts:
        print(f"  Testing with {num_states} states...")
        
        # Create model
        model = SimpleNSMModel(
            input_dim=input_dim,
            state_dim=state_dim,
            num_states=num_states,
            output_dim=output_dim
        )
        
        # Train model
        metrics = train_model(model, train_loader, epochs=epochs)
        
        # Evaluate model
        test_accuracy = evaluate_model(model, test_loader)
        
        # Store results
        results['state_counts'].append(num_states)
        results['accuracies'].append(metrics['accuracy'])
        results['test_accuracies'].append(test_accuracy)
        results['memory_usages'].append(metrics['memory_usage'])
        results['training_times'].append(metrics['training_time'])
        
        print(f"    Train Accuracy: {metrics['accuracy']:.2f}%")
        print(f"    Test Accuracy: {test_accuracy:.2f}%")
        print(f"    Memory Usage: {metrics['memory_usage']:.2f} MB")
        print(f"    Training Time: {metrics['training_time']:.2f} seconds")
    
    return results


if __name__ == "__main__":
    # Run the hyperparameter sweep
    print("Running hyperparameter sweep...")
    results = run_hyperparameter_sweep()
    
    # Convert results to JSON-serializable format
    serializable_results = {
        'state_counts': [int(x) for x in results['state_counts']],
        'accuracies': [float(x) for x in results['accuracies']],
        'test_accuracies': [float(x) for x in results['test_accuracies']],
        'memory_usages': [float(x) for x in results['memory_usages']],
        'training_times': [float(x) for x in results['training_times']]
    }
    
    # Save results
    with open('results/experiments/hyperparameter_sweep_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("Results saved to results/experiments/hyperparameter_sweep_results.json")
    print("Hyperparameter sweep completed successfully!")