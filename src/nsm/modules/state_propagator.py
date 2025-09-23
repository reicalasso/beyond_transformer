"""
State Propagator Module for Neural State Machines

This module implements a state propagator with gated updates and state-to-state communication.
"""

import torch
import torch.nn as nn


class StatePropagator(nn.Module):
    """
    State Propagator with LSTM/GRU-inspired gated updates and optional state-to-state communication.

    This module controls the update, retain, or reset behavior of state vectors using
    gating mechanisms inspired by LSTM and GRU architectures.
    """

    def __init__(
        self, state_dim, gate_type="gru", num_heads=4, enable_communication=True
    ):
        """
        Initialize the StatePropagator.

        Args:
            state_dim (int): Dimension of the state vectors
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
            num_heads (int): Number of attention heads for state-to-state communication
            enable_communication (bool): Whether to enable state-to-state communication
        """
        super(StatePropagator, self).__init__()
        self.state_dim = state_dim
        self.gate_type = gate_type
        self.enable_communication = enable_communication

        # Initialize gating mechanisms
        self._init_gates()

        # Initialize communication mechanism
        if enable_communication:
            self._init_communication(num_heads)

    def _init_gates(self):
        """Initialize the gating mechanisms based on gate_type."""
        gate_input_dim = self.state_dim * 2

        if self.gate_type == "gru":
            # GRU-style gates
            self.reset_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.update_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.candidate_state = nn.Linear(gate_input_dim, self.state_dim)
        elif self.gate_type == "lstm":
            # LSTM-style gates
            self.forget_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.input_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.output_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.candidate_state = nn.Linear(gate_input_dim, self.state_dim)
        else:
            raise ValueError(
                f"Unsupported gate_type: {self.gate_type}. Use 'lstm' or 'gru'."
            )

    def _init_communication(self, num_heads):
        """
        Initialize the state-to-state communication mechanism.

        Args:
            num_heads (int): Number of attention heads
        """
        self.attention = nn.MultiheadAttention(
            embed_dim=self.state_dim, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.state_dim)

    def forward(self, prev_state, new_input):
        """
        Apply gated update to propagate state.

        Args:
            prev_state (torch.Tensor): Previous state vector
                Shape: [batch_size, state_dim] or [batch_size, num_states, state_dim]
            new_input (torch.Tensor): New input vector
                Shape: [batch_size, state_dim] or [batch_size, num_states, state_dim]

        Returns:
            torch.Tensor: Updated state vector with same shape as input
        """
        # Handle both single state and multiple states
        if prev_state.dim() == 2:
            # Single state per batch
            updated_state = self._single_state_update(prev_state, new_input)
        else:
            # Multiple states per batch
            updated_state = self._multi_state_update(prev_state, new_input)

            # Apply state-to-state communication if enabled
            if self.enable_communication:
                updated_state = self._apply_communication(updated_state)

        return updated_state

    def _single_state_update(self, prev_state, new_input):
        """
        Update a single state vector.

        Args:
            prev_state (torch.Tensor): Previous state [batch_size, state_dim]
            new_input (torch.Tensor): New input [batch_size, state_dim]

        Returns:
            torch.Tensor: Updated state [batch_size, state_dim]
        """
        if self.gate_type == "gru":
            return self._gru_step(prev_state, new_input)
        else:  # lstm
            return self._lstm_step(prev_state, new_input)

    def _multi_state_update(self, prev_states, new_inputs):
        """
        Update multiple state vectors.

        Args:
            prev_states (torch.Tensor): Previous states [batch_size, num_states, state_dim]
            new_inputs (torch.Tensor): New inputs [batch_size, num_states, state_dim]

        Returns:
            torch.Tensor: Updated states [batch_size, num_states, state_dim]
        """
        if self.gate_type == "gru":
            return self._gru_step_multi(prev_states, new_inputs)
        else:  # lstm
            return self._lstm_step_multi(prev_states, new_inputs)

    def _gru_step(self, prev_state, new_input):
        """
        GRU-style gated update step for single state.

        Args:
            prev_state (torch.Tensor): Previous state vector
            new_input (torch.Tensor): New input vector

        Returns:
            torch.Tensor: Updated state vector
        """
        # Concatenate previous state and new input
        combined = torch.cat([prev_state, new_input], dim=1)

        # Calculate gates
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        update_gate = torch.sigmoid(self.update_gate(combined))

        # Calculate candidate state
        reset_prev_state = reset_gate * prev_state
        candidate_combined = torch.cat([reset_prev_state, new_input], dim=1)
        candidate_state = torch.tanh(self.candidate_state(candidate_combined))

        # Calculate new state using GRU formula
        new_state = (1 - update_gate) * prev_state + update_gate * candidate_state

        return new_state

    def _lstm_step(self, prev_state, new_input):
        """
        LSTM-style gated update step for single state.

        Args:
            prev_state (torch.Tensor): Previous state vector
            new_input (torch.Tensor): New input vector

        Returns:
            torch.Tensor: Updated state vector
        """
        # Concatenate previous state and new input
        combined = torch.cat([prev_state, new_input], dim=1)

        # Calculate gates
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))

        # Calculate candidate values
        candidate_state = torch.tanh(self.candidate_state(combined))

        # Calculate new cell state
        cell_state = forget_gate * prev_state + input_gate * candidate_state

        # Calculate new hidden state
        new_state = output_gate * torch.tanh(cell_state)

        return new_state

    def _gru_step_multi(self, prev_states, new_inputs):
        """
        GRU-style gated update step for multiple states.

        Args:
            prev_states (torch.Tensor): Previous state vectors [batch_size, num_states, state_dim]
            new_inputs (torch.Tensor): New input vectors [batch_size, num_states, state_dim]

        Returns:
            torch.Tensor: Updated state vectors [batch_size, num_states, state_dim]
        """
        batch_size, num_states, state_dim = prev_states.shape

        # Reshape for processing all states together
        prev_states_flat = prev_states.view(-1, state_dim)
        new_inputs_flat = new_inputs.view(-1, state_dim)

        # Concatenate previous states and new inputs
        combined = torch.cat([prev_states_flat, new_inputs_flat], dim=1)

        # Calculate gates
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        update_gate = torch.sigmoid(self.update_gate(combined))

        # Calculate candidate states
        reset_prev_states = reset_gate * prev_states_flat
        candidate_combined = torch.cat([reset_prev_states, new_inputs_flat], dim=1)
        candidate_states = torch.tanh(self.candidate_state(candidate_combined))

        # Calculate new states
        new_states = (
            1 - update_gate
        ) * prev_states_flat + update_gate * candidate_states

        # Reshape back to original dimensions
        new_states = new_states.view(batch_size, num_states, state_dim)

        return new_states

    def _lstm_step_multi(self, prev_states, new_inputs):
        """
        LSTM-style gated update step for multiple states.

        Args:
            prev_states (torch.Tensor): Previous state vectors [batch_size, num_states, state_dim]
            new_inputs (torch.Tensor): New input vectors [batch_size, num_states, state_dim]

        Returns:
            torch.Tensor: Updated state vectors [batch_size, num_states, state_dim]
        """
        batch_size, num_states, state_dim = prev_states.shape

        # Reshape for processing all states together
        prev_states_flat = prev_states.view(-1, state_dim)
        new_inputs_flat = new_inputs.view(-1, state_dim)

        # Concatenate previous states and new inputs
        combined = torch.cat([prev_states_flat, new_inputs_flat], dim=1)

        # Calculate gates
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))

        # Calculate candidate values
        candidate_states = torch.tanh(self.candidate_state(combined))

        # Calculate new cell states
        cell_states = forget_gate * prev_states_flat + input_gate * candidate_states

        # Calculate new hidden states
        new_states = output_gate * torch.tanh(cell_states)

        # Reshape back to original dimensions
        new_states = new_states.view(batch_size, num_states, state_dim)

        return new_states

    def _apply_communication(self, states):
        """
        Apply state-to-state communication using MultiHeadAttention.

        Args:
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]

        Returns:
            torch.Tensor: Updated state vectors [batch_size, num_states, state_dim]
        """
        # Apply self-attention
        attended_states, _ = self.attention(states, states, states)

        # Add residual connection and apply layer normalization
        updated_states = self.layer_norm(states + attended_states)

        return updated_states
