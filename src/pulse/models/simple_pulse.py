"""
PULSE Model Implementation

This module provides SimplePulse for single-input processing and
SequencePulse for sequence processing with temporal state updates.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Use relative import
from ..modules.state_propagator import StatePropagator


class SimplePulse(nn.Module):
    """
    A simple PULSE model for demonstration.

    This model shows how to use the StatePropagator in a complete model.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_states: int,
        output_dim: int,
        gate_type: str = "gru",
    ) -> None:
        """
        Initialize the SimplePulse model.

        Args:
            input_dim (int): Dimension of input features
            state_dim (int): Dimension of state vectors
            num_states (int): Number of state nodes
            output_dim (int): Dimension of output
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
        """
        super(SimplePulse, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.output_dim = output_dim
        self.gate_type = gate_type

        # Input projection to state dimension
        self.input_projection = nn.Linear(input_dim, state_dim)

        # State propagator
        self.state_propagator = StatePropagator(
            state_dim=state_dim, gate_type=gate_type, enable_communication=True
        )

        # Output projection
        self.output_projection = nn.Linear(state_dim * num_states, output_dim)

        # Initialize states
        self.initial_states = nn.Parameter(torch.randn(1, num_states, state_dim))

    def forward(
        self, x: torch.Tensor, return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, input_dim]
            return_states: If True, also return final states

        Returns:
            Output tensor of shape [batch_size, output_dim]
            If return_states=True, also returns states [batch_size, num_states, state_dim]
        """
        batch_size = x.size(0)

        # Project input to state dimension
        x = self.input_projection(x)  # [batch_size, state_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, state_dim]

        # Initialize states for batch
        states = self.initial_states.repeat(
            batch_size, 1, 1
        )  # [batch_size, num_states, state_dim]

        # Update states
        input_t = x.repeat(1, self.num_states, 1)  # [batch_size, num_states, state_dim]
        states = self.state_propagator(states, input_t)

        # Global pooling of states
        pooled_states = states.view(
            batch_size, -1
        )  # [batch_size, state_dim * num_states]

        # Output projection
        output = self.output_projection(pooled_states)

        if return_states:
            return output, states
        return output


class SequencePulse(nn.Module):
    """
    Sequence-aware PULSE model.

    Processes sequences by iterating through time steps and updating states.
    Suitable for language modeling, sequence classification, and time series.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_states: int,
        output_dim: int,
        gate_type: str = "gru",
        output_mode: str = "last",
    ) -> None:
        """
        Initialize the SequencePulse model.

        Args:
            input_dim: Dimension of input features per time step
            state_dim: Dimension of state vectors
            num_states: Number of state nodes
            output_dim: Dimension of output
            gate_type: Type of gating mechanism ('lstm' or 'gru')
            output_mode: How to produce output - 'last', 'all', or 'mean'
        """
        super(SequencePulse, self).__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.output_dim = output_dim
        self.gate_type = gate_type
        self.output_mode = output_mode

        # Input projection to state dimension
        self.input_projection = nn.Linear(input_dim, state_dim)

        # State propagator
        self.state_propagator = StatePropagator(
            state_dim=state_dim, gate_type=gate_type, enable_communication=True
        )

        # Output projection
        self.output_projection = nn.Linear(state_dim * num_states, output_dim)

        # Initialize states
        self.initial_states = nn.Parameter(torch.randn(1, num_states, state_dim) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
        return_all_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for sequence processing.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            initial_states: Optional initial states [batch_size, num_states, state_dim]
            return_all_states: If True, return states at all time steps

        Returns:
            Output tensor. Shape depends on output_mode:
                - 'last': [batch_size, output_dim]
                - 'all': [batch_size, seq_len, output_dim]
                - 'mean': [batch_size, output_dim]
            If return_all_states=True, also returns all states
        """
        batch_size, seq_len, _ = x.shape

        # Initialize states for batch
        if initial_states is not None:
            states = initial_states
        else:
            states = self.initial_states.repeat(
                batch_size, 1, 1
            )  # [batch_size, num_states, state_dim]

        # Process sequence step by step
        all_outputs = []
        all_states = [] if return_all_states else None

        for t in range(seq_len):
            # Project input at time t
            input_t = self.input_projection(x[:, t, :])  # [batch_size, state_dim]
            input_t = input_t.unsqueeze(1).repeat(
                1, self.num_states, 1
            )  # [batch_size, num_states, state_dim]

            # Update states
            states = self.state_propagator(states, input_t)

            if return_all_states:
                all_states.append(states.clone())

            # Compute output for this time step
            pooled_states = states.view(
                batch_size, -1
            )  # [batch_size, state_dim * num_states]
            output_t = self.output_projection(pooled_states)  # [batch_size, output_dim]
            all_outputs.append(output_t)

        # Stack outputs
        all_outputs = torch.stack(all_outputs, dim=1)  # [batch_size, seq_len, output_dim]

        # Apply output mode
        if self.output_mode == "last":
            output = all_outputs[:, -1, :]  # [batch_size, output_dim]
        elif self.output_mode == "mean":
            output = all_outputs.mean(dim=1)  # [batch_size, output_dim]
        else:  # 'all'
            output = all_outputs  # [batch_size, seq_len, output_dim]

        if return_all_states:
            all_states = torch.stack(all_states, dim=1)  # [batch_size, seq_len, num_states, state_dim]
            return output, all_states
        return output

    def get_state_info(self) -> dict:
        """Get information about the model's state configuration."""
        return {
            "num_states": self.num_states,
            "state_dim": self.state_dim,
            "gate_type": self.gate_type,
            "output_mode": self.output_mode,
        }


# Example usage
if __name__ == "__main__":
    # Create model
    model = SimplePulse(
        input_dim=784,  # MNIST-like input
        state_dim=128,  # State dimension
        num_states=16,  # Number of states
        output_dim=10,  # Classification output
        gate_type="gru",  # Gating mechanism
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

    # Test SequencePulse
    print("\n--- Testing SequencePulse ---")
    seq_model = SequencePulse(
        input_dim=64,
        state_dim=128,
        num_states=16,
        output_dim=10,
        gate_type="gru",
        output_mode="last",
    )

    # Create sequence input
    seq_x = torch.randn(batch_size, 20, 64)  # [batch, seq_len, input_dim]
    seq_output = seq_model(seq_x)

    print(f"Sequence input shape: {seq_x.shape}")
    print(f"Sequence output shape: {seq_output.shape}")

    # Test with return_all_states
    seq_output, all_states = seq_model(seq_x, return_all_states=True)
    print(f"All states shape: {all_states.shape}")

    # Test differentiability
    loss = seq_output.sum()
    loss.backward()
    print("SequencePulse backward pass successful!")