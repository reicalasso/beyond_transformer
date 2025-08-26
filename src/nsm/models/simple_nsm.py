"""
Example NSM Model Implementation
"""

import torch
import torch.nn as nn
from nsm.modules import StatePropagator


class SimpleNSM(nn.Module):
    """
    A simple Neural State Machine model for demonstration.
    
    This model shows how to use the StatePropagator in a complete model.
    """
    
    def __init__(self, input_dim, state_dim, num_states, output_dim, gate_type='gru'):
        """
        Initialize the SimpleNSM model.
        
        Args:
            input_dim (int): Dimension of input features
            state_dim (int): Dimension of state vectors
            num_states (int): Number of state nodes
            output_dim (int): Dimension of output
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
        """
        super(SimpleNSM, self).__init__()
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
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Project input to state dimension
        x = self.input_projection(x)  # [batch_size, state_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, state_dim]
        
        # Initialize states for batch
        states = self.initial_states.repeat(batch_size, 1, 1)  # [batch_size, num_states, state_dim]
        
        # Update states
        input_t = x.repeat(1, self.num_states, 1)  # [batch_size, num_states, state_dim]
        states = self.state_propagator(states, input_t)
        
        # Global pooling of states
        pooled_states = states.view(batch_size, -1)  # [batch_size, state_dim * num_states]
        
        # Output projection
        output = self.output_projection(pooled_states)
        
        return output


# Example usage
if __name__ == "__main__":
    # Create model
    model = SimpleNSM(
        input_dim=784,      # MNIST-like input
        state_dim=128,      # State dimension
        num_states=16,      # Number of states
        output_dim=10,      # Classification output
        gate_type='gru'     # Gating mechanism
    )
    
    # Create sample input
    batch_size = 32
    x = torch.randn(batch_size, 784)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model forward pass successful!")
    
    # Test differentiability
    loss = output.sum()
    loss.backward()
    print("Backward pass successful - model is differentiable!")