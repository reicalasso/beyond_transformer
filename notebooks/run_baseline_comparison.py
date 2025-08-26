#!/usr/bin/env python3
"""
Extended Baseline Experiments

This script compares Neural State Machines (NSM) against various baseline architectures:
- Transformers
- LSTM/GRU
- RWKV (conceptual)
- S4 (conceptual)

Metrics: Accuracy, F1, Memory, FLOPs, Training Speed
Datasets: MNIST, Tiny Shakespeare, IMDb, CIFAR-10, LRA (conceptual)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import psutil
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our components
from nsm.models import SimpleNSM


class SimpleLSTM(nn.Module):
    """Simple LSTM baseline model."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        if x.dim() == 2:
            # Reshape for LSTM: [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
            
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class SimpleGRU(nn.Module):
    """Simple GRU baseline model."""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(SimpleGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        if x.dim() == 2:
            # Reshape for GRU: [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
            
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class SimpleTransformer(nn.Module):
    """Simple Transformer baseline model."""
    
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.0):
        super(SimpleTransformer, self).__init__()
        self.model_dim = model_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1000, model_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(model_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        if x.dim() == 2:
            # Reshape for Transformer: [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = self.input_embedding(x)
        
        # Add positional encoding
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + pos_enc
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


def create_mnist_dataset(num_samples=1000):
    """Create MNIST-like dataset."""
    # MNIST: 28x28 = 784 pixels, 10 classes
    X = torch.randn(num_samples, 784)
    y = torch.randint(0, 10, (num_samples,))
    # Reshape for sequence models
    X = X.unsqueeze(1)  # Add sequence dimension: [batch, 1, 784]
    return torch.utils.data.TensorDataset(X, y)


def create_tiny_shakespeare_dataset(num_samples=1000, seq_len=256):
    """Create Tiny Shakespeare-like dataset."""
    # Vocabulary size for characters (simplified)
    vocab_size = 100
    X = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    # For language modeling, predict next character
    y = torch.randint(0, vocab_size, (num_samples,), dtype=torch.long)
    # Keep as is for text models (no reshaping needed)
    return torch.utils.data.TensorDataset(X, y)


def create_imdb_dataset(num_samples=1000, seq_len=512):
    """Create IMDb-like dataset."""
    # Vocabulary size for words (simplified)
    vocab_size = 10000
    X = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
    # Binary sentiment classification
    y = torch.randint(0, 2, (num_samples,))
    # Keep as is for text models (no reshaping needed)
    return torch.utils.data.TensorDataset(X, y)


def create_cifar10_dataset(num_samples=1000):
    """Create CIFAR-10-like dataset."""
    # CIFAR-10: 32x32x3 = 3072 pixels, 10 classes
    X = torch.randn(num_samples, 3072)
    y = torch.randint(0, 10, (num_samples,))
    # Reshape for sequence models
    X = X.unsqueeze(1)  # Add sequence dimension: [batch, 1, 3072]
    return torch.utils.data.TensorDataset(X, y)


def train_model(model, train_loader, epochs=3, lr=0.001, weight_decay=1e-5):
    """Train model and return metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Memory tracking
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Training time tracking
    start_time = time.time()
    
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Handle discrete data (embedding for text)
            if data.dtype == torch.int64 and len(data.shape) == 2:
                # Text data - add embedding layer to model if not exists
                if not hasattr(model, 'embedding'):
                    # Determine vocab size based on data
                    vocab_size = data.max().item() + 1
                    # Ensure minimum vocab size
                    vocab_size = max(vocab_size, 100)
                    model.embedding = nn.Embedding(vocab_size, 128).to(device)
                data = model.embedding(data)
            elif data.dim() == 3 and data.shape[1] == 1:
                # Squeeze single sequence dimension for non-text data
                data = data.squeeze(1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Limit batches for quick testing
            if batch_idx > 20:
                break
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    
    end_time = time.time()
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate metrics
    training_time = end_time - start_time
    memory_usage = final_memory - initial_memory
    final_accuracy = epoch_accuracies[-1]
    
    return {
        'accuracy': final_accuracy,
        'losses': epoch_losses,
        'accuracies': epoch_accuracies,
        'memory_usage': memory_usage,
        'training_time': training_time
    }


def evaluate_model(model, test_loader):
    """Evaluate model and return metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Handle discrete data (embedding for text)
            if data.dtype == torch.int64 and len(data.shape) == 2:
                # Text data - ensure embedding layer exists
                if not hasattr(model, 'embedding'):
                    # Determine vocab size based on data
                    vocab_size = data.max().item() + 1
                    # Ensure minimum vocab size
                    vocab_size = max(vocab_size, 100)
                    model.embedding = nn.Embedding(vocab_size, 128).to(device)
                data = model.embedding(data)
            elif data.dim() == 3 and data.shape[1] == 1:
                # Squeeze single sequence dimension for non-text data
                data = data.squeeze(1)
            
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Limit for quick testing
            if total > 200:
                break
    
    accuracy = 100. * correct / total
    return accuracy


def create_model(model_type, input_dim, output_dim, is_text=False):
    """Create model based on type."""
    
    if model_type == "LSTM":
        # For text data, input_dim is vocab size, so we use embedding
        if is_text:
            # Return a model that handles text data with embedding
            class TextLSTM(nn.Module):
                def __init__(self, vocab_size, hidden_dim, num_layers, output_dim, dropout=0.0):
                    super(TextLSTM, self).__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.embedding = nn.Embedding(vocab_size, hidden_dim)
                    self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, 
                                       batch_first=True, dropout=dropout)
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_dim, output_dim)
                
                def forward(self, x):
                    # x shape: [batch_size, seq_len] (token indices)
                    x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
                    batch_size = x.size(0)
                    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                    c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.dropout(out[:, -1, :])
                    out = self.fc(out)
                    return out
            
            return TextLSTM(input_dim, 128, 2, output_dim)
        else:
            return SimpleLSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=output_dim)
    
    elif model_type == "GRU":
        # For text data, input_dim is vocab size, so we use embedding
        if is_text:
            # Return a model that handles text data with embedding
            class TextGRU(nn.Module):
                def __init__(self, vocab_size, hidden_dim, num_layers, output_dim, dropout=0.0):
                    super(TextGRU, self).__init__()
                    self.hidden_dim = hidden_dim
                    self.num_layers = num_layers
                    self.embedding = nn.Embedding(vocab_size, hidden_dim)
                    self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, 
                                     batch_first=True, dropout=dropout)
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_dim, output_dim)
                
                def forward(self, x):
                    # x shape: [batch_size, seq_len] (token indices)
                    x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
                    batch_size = x.size(0)
                    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                    out, _ = self.gru(x, h0)
                    out = self.dropout(out[:, -1, :])
                    out = self.fc(out)
                    return out
            
            return TextGRU(input_dim, 128, 2, output_dim)
        else:
            return SimpleGRU(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=output_dim)
    
    elif model_type == "Transformer":
        # For text data, input_dim is vocab size, so we use embedding
        if is_text:
            # Return a model that handles text data with embedding
            class TextTransformer(nn.Module):
                def __init__(self, vocab_size, model_dim, num_heads, num_layers, output_dim, dropout=0.0):
                    super(TextTransformer, self).__init__()
                    self.model_dim = model_dim
                    self.embedding = nn.Embedding(vocab_size, model_dim)
                    self.pos_encoding = nn.Parameter(torch.randn(1000, model_dim))
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=model_dim, 
                        nhead=num_heads, 
                        dropout=dropout,
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(model_dim, output_dim)
                
                def forward(self, x):
                    # x shape: [batch_size, seq_len] (token indices)
                    batch_size, seq_len = x.shape
                    x = self.embedding(x)  # [batch_size, seq_len, model_dim]
                    pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
                    x = x + pos_enc
                    x = self.transformer(x)
                    x = x.mean(dim=1)  # Global average pooling
                    x = self.dropout(x)
                    x = self.fc(x)
                    return x
            
            return TextTransformer(input_dim, 128, 4, 2, output_dim)
        else:
            return SimpleTransformer(input_dim=input_dim, model_dim=128, num_heads=4, 
                                   num_layers=2, output_dim=output_dim)
    
    elif model_type == "NSM":
        # For text data, we need to handle it specially
        if is_text:
            # Return a model that handles text data with embedding
            class TextNSM(nn.Module):
                def __init__(self, vocab_size, state_dim, num_states, output_dim):
                    super(TextNSM, self).__init__()
                    self.embedding = nn.Embedding(vocab_size, state_dim)
                    self.nsm = SimpleNSM(input_dim=state_dim, state_dim=state_dim, 
                                       num_states=num_states, output_dim=output_dim)
                
                def forward(self, x):
                    # x shape: [batch_size, seq_len] (token indices)
                    x = self.embedding(x)  # [batch_size, seq_len, state_dim]
                    return self.nsm(x)
            
            return TextNSM(input_dim, 64, 16, output_dim)
        else:
            return SimpleNSM(input_dim=input_dim, state_dim=64, num_states=16, output_dim=output_dim)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_baseline_comparison():
    """Run baseline comparison experiment."""
    
    # Model types to compare
    model_types = ["LSTM", "GRU", "Transformer", "NSM"]
    
    # Dataset configurations
    datasets = {
        'MNIST': {
            'create_func': create_mnist_dataset,
            'input_dim': 784,
            'output_dim': 10,
            'num_samples': 1000
        },
        'Tiny_Shakespeare': {
            'create_func': create_tiny_shakespeare_dataset,
            'input_dim': 100,  # vocab_size
            'output_dim': 100,  # vocab_size
            'num_samples': 500
        },
        'IMDb': {
            'create_func': create_imdb_dataset,
            'input_dim': 10000,  # vocab_size
            'output_dim': 2,     # binary classification
            'num_samples': 500
        },
        'CIFAR10': {
            'create_func': create_cifar10_dataset,
            'input_dim': 3072,
            'output_dim': 10,
            'num_samples': 1000
        }
    }
    
    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Experiment parameters
    epochs = 3
    batch_size = 32
    
    for dataset_name, dataset_config in datasets.items():
        print(f"\n=== Testing {dataset_name} ===")
        
        # Create dataset
        train_dataset = dataset_config['create_func'](num_samples=dataset_config['num_samples'])
        test_dataset = dataset_config['create_func'](num_samples=200)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        for model_type in model_types:
            print(f"  Testing {model_type}...")
            
            # Check if this is a text dataset
            is_text = dataset_name in ['Tiny_Shakespeare', 'IMDb']
            
            try:
                # Create model
                model = create_model(
                    model_type=model_type,
                    input_dim=dataset_config['input_dim'],
                    output_dim=dataset_config['output_dim'],
                    is_text=is_text
                )
                
                # Train model
                metrics = train_model(model, train_loader, epochs=epochs)
                
                # Evaluate model
                test_accuracy = evaluate_model(model, test_loader)
                
                # Store results
                results[dataset_name][model_type] = {
                    'train_accuracy': metrics['accuracy'],
                    'test_accuracy': test_accuracy,
                    'memory_usage': metrics['memory_usage'],
                    'training_time': metrics['training_time'],
                    'losses': metrics['losses'],
                    'accuracies': metrics['accuracies']
                }
                
                print(f"    Train Accuracy: {metrics['accuracy']:.2f}%")
                print(f"    Test Accuracy: {test_accuracy:.2f}%")
                print(f"    Memory Usage: {metrics['memory_usage']:.2f} MB")
                print(f"    Training Time: {metrics['training_time']:.2f} seconds")
                
            except Exception as e:
                print(f"    Error with {model_type}: {e}")
                results[dataset_name][model_type] = {
                    'train_accuracy': 0.0,
                    'test_accuracy': 0.0,
                    'memory_usage': 0.0,
                    'training_time': 0.0,
                    'losses': [],
                    'accuracies': [],
                    'error': str(e)
                }
    
    return results


def plot_comparison_results(results):
    """Plot comparison results."""
    
    model_types = ["LSTM", "GRU", "Transformer", "NSM"]
    datasets = list(results.keys())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Baseline Model Comparison Results', fontsize=16)
    
    # Plot 1: Test Accuracy
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, model_type in enumerate(model_types):
        accuracies = [results[dataset][model_type].get('test_accuracy', 0) for dataset in datasets]
        ax.bar(x + i*width, accuracies, width, label=model_type)
    
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy by Model and Dataset')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Time
    ax = axes[0, 1]
    for i, model_type in enumerate(model_types):
        times = [results[dataset][model_type].get('training_time', 0) for dataset in datasets]
        ax.bar(x + i*width, times, width, label=model_type)
    
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time by Model and Dataset')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Memory Usage
    ax = axes[1, 0]
    for i, model_type in enumerate(model_types):
        memory = [results[dataset][model_type].get('memory_usage', 0) for dataset in datasets]
        ax.bar(x + i*width, memory, width, label=model_type)
    
    ax.set_xlabel('Datasets')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Usage by Model and Dataset')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Curves (example for first dataset)
    ax = axes[1, 1]
    if datasets:
        first_dataset = datasets[0]
        for model_type in model_types:
            accuracies = results[first_dataset][model_type].get('accuracies', [])
            if accuracies:
                ax.plot(accuracies, marker='o', label=model_type)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Accuracy (%)')
        ax.set_title(f'Training Curves - {first_dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results, filename='baseline_comparison_results.json'):
    """Save results to JSON file."""
    # Convert results to JSON-serializable format
    serializable_results = {}
    for dataset_name, models in results.items():
        serializable_results[dataset_name] = {}
        for model_type, metrics in models.items():
            serializable_results[dataset_name][model_type] = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    serializable_results[dataset_name][model_type][key] = value
                elif isinstance(value, list):
                    serializable_results[dataset_name][model_type][key] = [float(x) if isinstance(x, (int, float)) else str(x) for x in value]
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("Running Extended Baseline Experiments")
    print("=" * 50)
    
    # Run the baseline comparison
    results = run_baseline_comparison()
    
    # Plot the results
    plot_comparison_results(results)
    
    # Save results
    save_results(results)
    
    print("\nBaseline comparison experiment completed!")
    print("Results saved to baseline_comparison_results.json")
    print("Plot saved to baseline_comparison_results.png")