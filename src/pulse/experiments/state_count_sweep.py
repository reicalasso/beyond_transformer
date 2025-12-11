"""
Hyperparameter Sweep Experiment for State Count

This module implements a hyperparameter sweep to test different numbers of state nodes
(8, 16, 32, 64) and measures accuracy, memory usage, and training speed.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from .experiment_results import ExperimentResults
from pulse.models import SimplePulse


def create_mnist_dataset(num_samples=1000):
    """
    Create MNIST-like synthetic dataset.

    Args:
        num_samples (int): Number of samples to generate

    Returns:
        torch.utils.data.TensorDataset: Generated dataset
    """
    # MNIST: 28x28 = 784 pixels, 10 classes
    X = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    return torch.utils.data.TensorDataset(X, y)


def train_model(model, train_loader, epochs=3, lr=0.001):
    """
    Train the model and return metrics.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        epochs (int): Number of epochs
        lr (float): Learning rate

    Returns:
        dict: Training metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if batch_idx > 10:
                break

    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Calculate metrics
    training_time = end_time - start_time
    memory_usage = final_memory - initial_memory
    accuracy = 100.0 * correct / total

    return {
        "accuracy": accuracy,
        "memory_usage": memory_usage,
        "training_time": training_time,
    }


def evaluate_model(model, test_loader):
    """
    Evaluate the model and return accuracy.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader

    Returns:
        float: Test accuracy percentage
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            if total > 100:
                break

    accuracy = 100.0 * correct / total
    return accuracy


def run_state_count_sweep():
    """
    Run hyperparameter sweep for different state counts.

    Returns:
        dict: Results for different state counts
    """
    # State counts to test
    state_counts = [8, 16, 32, 64]

    # Results storage
    results = {
        "state_counts": [],
        "accuracies": [],
        "test_accuracies": [],
        "memory_usages": [],
        "training_times": [],
    }

    # Fixed parameters
    input_dim = 784
    state_dim = 128
    output_dim = 10
    epochs = 3
    batch_size = 32

    print("Running State Count Hyperparameter Sweep...")
    print("=" * 50)

    # Create dataset
    train_dataset = create_mnist_dataset(num_samples=500)
    test_dataset = create_mnist_dataset(num_samples=100)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    for num_states in state_counts:
        print(f"Testing with {num_states} states...")

        # Create model
        model = SimplePulse(
            input_dim=input_dim,
            state_dim=state_dim,
            num_states=num_states,
            output_dim=output_dim,
            gate_type="gru",
        )

        # Train model
        metrics = train_model(model, train_loader, epochs=epochs)

        # Evaluate model
        test_accuracy = evaluate_model(model, test_loader)

        # Store results
        results["state_counts"].append(num_states)
        results["accuracies"].append(metrics["accuracy"])
        results["test_accuracies"].append(test_accuracy)
        results["memory_usages"].append(metrics["memory_usage"])
        results["training_times"].append(metrics["training_time"])

        print(f"  Train Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Memory Usage: {metrics['memory_usage']:.2f} MB")
        print(f"  Training Time: {metrics['training_time']:.2f} seconds")
        print()

    return results


def main():
    """Main function to run the state count sweep experiment."""
    # Create experiment results manager
    results_manager = ExperimentResults()

    # Run the hyperparameter sweep
    results = run_state_count_sweep()

    # Save results using the results manager
    filepath = results_manager.save_experiment_results("state_count_sweep", results)
    print(f"Results saved to {filepath}")

    print("Hyperparameter sweep completed successfully!")


if __name__ == "__main__":
    main()
