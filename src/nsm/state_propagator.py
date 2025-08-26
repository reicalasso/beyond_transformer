import torch
import torch.nn as nn
import torch.nn.functional as F


class StatePropagator(nn.Module):
    """
    StatePropagator with LSTM/GRU-inspired gated updates for controlling
    update, retain, or reset behavior of state vectors.
    """
    
    def __init__(self, state_dim, gate_type='gru', num_heads=4, enable_communication=True):
        """
        Initialize StatePropagator with gated mechanisms.
        
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
        
        # For GRU-style gating
        if gate_type == 'gru':
            # Reset gate: determines how much past state to forget
            self.reset_gate = nn.Linear(state_dim * 2, state_dim)
            # Update gate: determines how much new information to add
            self.update_gate = nn.Linear(state_dim * 2, state_dim)
            # Candidate state calculation
            self.candidate_state = nn.Linear(state_dim * 2, state_dim)
        # For LSTM-style gating
        elif gate_type == 'lstm':
            # Forget gate: determines what information to discard from cell state
            self.forget_gate = nn.Linear(state_dim * 2, state_dim)
            # Input gate: determines what new information to store in cell state
            self.input_gate = nn.Linear(state_dim * 2, state_dim)
            # Output gate: determines what to output based on cell state
            self.output_gate = nn.Linear(state_dim * 2, state_dim)
            # Candidate values to add to state
            self.candidate_state = nn.Linear(state_dim * 2, state_dim)
        else:
            raise ValueError("gate_type must be 'lstm' or 'gru'")
            
        # State-to-state communication using MultiHeadAttention
        if enable_communication:
            self.attention = nn.MultiheadAttention(
                embed_dim=state_dim, 
                num_heads=num_heads, 
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(state_dim)
        
    def forward(self, prev_state, new_input):
        """
        Apply gated update to propagate state.
        
        Args:
            prev_state (torch.Tensor): Previous state vector [batch_size, state_dim] or [batch_size, num_states, state_dim]
            new_input (torch.Tensor): New input vector [batch_size, state_dim] or [batch_size, num_states, state_dim]
            
        Returns:
            torch.Tensor: Updated state vector [batch_size, state_dim] or [batch_size, num_states, state_dim]
        """
        # Handle both single state and multiple states
        if prev_state.dim() == 2:  # Single state per batch
            if self.gate_type == 'gru':
                updated_state = self._gru_step(prev_state, new_input)
            else:  # lstm
                updated_state = self._lstm_step(prev_state, new_input)
        else:  # Multiple states per batch
            if self.gate_type == 'gru':
                updated_state = self._gru_step_multi(prev_state, new_input)
            else:  # lstm
                updated_state = self._lstm_step_multi(prev_state, new_input)
            
            # Apply state-to-state communication if enabled
            if self.enable_communication:
                updated_state = self._apply_communication(updated_state)
        
        return updated_state
    
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
        
        # Calculate new state
        new_state = (1 - update_gate) * prev_state + update_gate * candidate_state
        
        return new_state
    
    def _lstm_step(self, prev_state, new_input):
        """
        LSTM-style gated update step for single state.
        
        Args:
            prev_state (torch.Tensor): Previous state vector (acting as cell state)
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
        prev_states_flat = prev_states.view(-1, state_dim)  # [batch_size * num_states, state_dim]
        new_inputs_flat = new_inputs.view(-1, state_dim)    # [batch_size * num_states, state_dim]
        
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
        new_states = (1 - update_gate) * prev_states_flat + update_gate * candidate_states
        
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
        prev_states_flat = prev_states.view(-1, state_dim)  # [batch_size * num_states, state_dim]
        new_inputs_flat = new_inputs.view(-1, state_dim)    # [batch_size * num_states, state_dim]
        
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
            torch.Tensor: Updated state vectors after communication [batch_size, num_states, state_dim]
        """
        # Apply self-attention
        attended_states, _ = self.attention(states, states, states)
        
        # Add residual connection and apply layer normalization
        updated_states = self.layer_norm(states + attended_states)
        
        return updated_states


# Example usage
if __name__ == "__main__":
    # Test the StatePropagator with single state
    batch_size = 4
    state_dim = 128
    
    # Create propagator for single state
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    # Create sample inputs
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    # Forward pass
    updated_state = propagator(prev_state, new_input)
    
    print(f"Previous state shape: {prev_state.shape}")
    print(f"New input shape: {new_input.shape}")
    print(f"Updated state shape: {updated_state.shape}")
    
    # Test with multiple states and communication
    num_states = 8
    
    # Create propagator for multiple states
    propagator_multi = StatePropagator(state_dim, gate_type='gru', enable_communication=True)
    
    # Create sample inputs for multiple states
    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)
    
    # Forward pass
    updated_states = propagator_multi(prev_states, new_inputs)
    
    print(f"\nMultiple states test:")
    print(f"Previous states shape: {prev_states.shape}")
    print(f"New inputs shape: {new_inputs.shape}")
    print(f"Updated states shape: {updated_states.shape}")
    
    # Verify differentiability
    loss = updated_states.sum()
    loss.backward()
    print("Backward pass successful - module is differentiable")