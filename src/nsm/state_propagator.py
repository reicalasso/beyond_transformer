import torch
import torch.nn as nn
import torch.nn.functional as F


class StatePropagator(nn.Module):
    """
    StatePropagator with LSTM/GRU-inspired gated updates for controlling
    update, retain, or reset behavior of state vectors.
    """
    
    def __init__(self, state_dim, gate_type='gru'):
        """
        Initialize StatePropagator with gated mechanisms.
        
        Args:
            state_dim (int): Dimension of the state vectors
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
        """
        super(StatePropagator, self).__init__()
        self.state_dim = state_dim
        self.gate_type = gate_type
        
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
    
    def forward(self, prev_state, new_input):
        """
        Apply gated update to propagate state.
        
        Args:
            prev_state (torch.Tensor): Previous state vector [batch_size, state_dim]
            new_input (torch.Tensor): New input vector [batch_size, state_dim]
            
        Returns:
            torch.Tensor: Updated state vector [batch_size, state_dim]
        """
        if self.gate_type == 'gru':
            return self._gru_step(prev_state, new_input)
        else:  # lstm
            return self._lstm_step(prev_state, new_input)
    
    def _gru_step(self, prev_state, new_input):
        """
        GRU-style gated update step.
        
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
        LSTM-style gated update step.
        
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


# Example usage
if __name__ == "__main__":
    # Test the StatePropagator
    batch_size = 4
    state_dim = 128
    
    # Create propagator
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    # Create sample inputs
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    # Forward pass
    updated_state = propagator(prev_state, new_input)
    
    print(f"Previous state shape: {prev_state.shape}")
    print(f"New input shape: {new_input.shape}")
    print(f"Updated state shape: {updated_state.shape}")
    
    # Verify differentiability
    loss = updated_state.sum()
    loss.backward()
    print("Backward pass successful - module is differentiable")