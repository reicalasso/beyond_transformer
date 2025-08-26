"""
Experiment: Dynamic State Allocation and Pruning

This experiment evaluates the effect of dynamic state allocation and pruning
on memory usage and model performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict

from nsm import StateManager, NSMLayer, TokenToStateRouter


class DynamicNSM(nn.Module):
    """
    Neural State Machine with dynamic state allocation and pruning.
    """
    
    def __init__(self, input_dim, state_dim, max_states, output_dim):
        """
        Initialize the DynamicNSM.
        
        Args:
            input_dim (int): Input dimension
            state_dim (int): State dimension
            max_states (int): Maximum number of states
            output_dim (int): Output dimension
        """
        super(DynamicNSM, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.max_states = max_states
        self.output_dim = output_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, state_dim)
        
        # State manager with dynamic allocation
        self.state_manager = StateManager(
            state_dim=state_dim,
            max_states=max_states,
            initial_states=8,
            prune_threshold=0.3
        )
        
        # NSM layer
        self.nsm_layer = NSMLayer(state_dim=state_dim, token_dim=state_dim, num_heads=4)
        
        # Output projection
        self.output_projection = nn.Linear(state_dim * 8, output_dim)  # Fixed number for simplicity
        
        # Initialize state memory
        self.state_memory = nn.Parameter(torch.randn(1, max_states, state_dim))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Embed input
        embedded = self.input_embedding(x)  # [batch_size, state_dim]
        embedded = embedded.unsqueeze(1)     # [batch_size, 1, state_dim]
        
        # Get current states
        current_states = self.state_memory[:, :self.state_manager.get_active_count(), :]
        current_states = current_states.repeat(batch_size, 1, 1)  # [batch_size, active_states, state_dim]
        
        # Expand embedded input to match states
        expanded_input = embedded.repeat(1, current_states.size(1), 1)  # [batch_size, active_states, state_dim]
        
        # Apply NSM layer
        updated_states = self.nsm_layer(current_states, expanded_input)
        
        # Global pooling
        pooled_states = updated_states.view(batch_size, -1)
        
        # Output projection
        output = self.output_projection(pooled_states)
        
        return output
    
    def prune_states(self):
        """
        Prune low importance states.
        
        Returns:
            int: Number of states pruned
        """
        return self.state_manager.prune_low_importance_states()
    
    def allocate_states(self, num_states):
        """
        Allocate new states.
        
        Args:
            num_states (int): Number of states to allocate
            
        Returns:
            int: Number of states allocated
        """
        return self.state_manager.allocate_states(num_states)


def create_synthetic_data(num_samples, input_dim, output_dim):
    """
    Create synthetic dataset for testing.
    
    Args:
        num_samples (int): Number of samples
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        
    Returns:
        torch.utils.data.TensorDataset: Synthetic dataset
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    return torch.utils.data.TensorDataset(X, y)


def train_model(model, train_loader, epochs=5, lr=0.001):
    """
    Train the model and track state dynamics.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        epochs (int): Number of epochs
        lr (float): Learning rate
        
    Returns:
        dict: Training metrics and state dynamics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    metrics = {
        'losses': [],
        'accuracies': [],
        'active_states': [],
        'pruned_states': [],
        'allocated_states': []
    }
    
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
            
            # Periodic state management
            if batch_idx % 10 == 0:
                # Prune states
                pruned = model.prune_states()
                metrics['pruned_states'].append(pruned)
                
                # Allocate states if needed
                if pruned > 0:
                    allocated = model.allocate_states(min(pruned, 2))
                    metrics['allocated_states'].append(allocated)
                else:
                    metrics['allocated_states'].append(0)
        
        # Record metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        active_states = model.state_manager.get_active_count()
        
        metrics['losses'].append(avg_loss)
        metrics['accuracies'].append(accuracy)
        metrics['active_states'].append(active_states)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
              f"Accuracy={accuracy:.2f}%, Active States={active_states}")
    
    return metrics


def run_dynamic_state_experiment():
    """
    Run the dynamic state allocation and pruning experiment.
    """
    print("Running Dynamic State Allocation and Pruning Experiment")
    print("=" * 60)
    
    # Experiment parameters
    input_dim = 128
    state_dim = 64
    max_states = 32
    output_dim = 10
    num_samples = 1000
    
    # Create model
    model = DynamicNSM(
        input_dim=input_dim,
        state_dim=state_dim,
        max_states=max_states,
        output_dim=output_dim
    )
    
    # Create dataset
    dataset = create_synthetic_data(num_samples, input_dim, output_dim)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    start_time = time.time()
    metrics = train_model(model, train_loader, epochs=5)
    end_time = time.time()
    
    # Print results
    print("\nExperiment Results:")
    print("-" * 30)
    print(f"Training time: {end_time - start_time:.2f} seconds")
    print(f"Final accuracy: {metrics['accuracies'][-1]:.2f}%")
    print(f"Initial active states: {metrics['active_states'][0]}")
    print(f"Final active states: {metrics['active_states'][-1]}")
    print(f"Total states pruned: {sum(metrics['pruned_states'])}")
    print(f"Total states allocated: {sum(metrics['allocated_states'])}")
    
    # Test state manager info
    state_info = model.state_manager.get_state_info()
    print(f"\nFinal state information:")
    print(f"  Total states: {state_info['total_states']}")
    print(f"  Active states: {state_info['active_states']}")
    print(f"  Importance scores range: "
          f"{min(state_info['importance_scores']):.3f} - "
          f"{max(state_info['importance_scores']):.3f}")
    
    return metrics


if __name__ == "__main__":
    metrics = run_dynamic_state_experiment()
    print("\n✓ Dynamic state allocation experiment completed!")