"""
PULSE State Management

Provides state propagation and management:
- GatedStatePropagator: GRU/LSTM-style state updates
- StateManager: Dynamic state allocation and pruning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedStatePropagator(nn.Module):
    """
    Gated State Propagator with GRU-style updates.
    
    Updates states based on new information using gating mechanisms.
    
    Args:
        state_dim: State dimension
        gate_type: 'gru' or 'lstm'
    """
    
    def __init__(
        self,
        state_dim: int,
        gate_type: str = "gru",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.gate_type = gate_type
        
        if gate_type == "gru":
            # Fused gates: [reset, update, candidate]
            self.gate_proj = nn.Linear(state_dim * 2, state_dim * 3, bias=True)
        elif gate_type == "lstm":
            # Fused gates: [forget, input, output, candidate]
            self.gate_proj = nn.Linear(state_dim * 2, state_dim * 4, bias=True)
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def forward(
        self,
        states: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update states with new inputs.
        
        Args:
            states: Current states [batch, num_states, state_dim]
            inputs: New information [batch, num_states, state_dim]
            
        Returns:
            Updated states [batch, num_states, state_dim]
        """
        combined = torch.cat([states, inputs], dim=-1)
        
        if self.gate_type == "gru":
            return self._gru_update(states, combined)
        else:
            return self._lstm_update(states, combined)
    
    def _gru_update(self, states: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        """GRU-style update."""
        gates = self.gate_proj(combined)
        reset, update, candidate = gates.chunk(3, dim=-1)
        
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(candidate * reset)
        
        return (1 - update) * states + update * candidate
    
    def _lstm_update(self, states: torch.Tensor, combined: torch.Tensor) -> torch.Tensor:
        """LSTM-style update."""
        gates = self.gate_proj(combined)
        forget, input_gate, output, candidate = gates.chunk(4, dim=-1)
        
        forget = torch.sigmoid(forget)
        input_gate = torch.sigmoid(input_gate)
        output = torch.sigmoid(output)
        candidate = torch.tanh(candidate)
        
        cell = forget * states + input_gate * candidate
        return output * torch.tanh(cell)


class StateManager(nn.Module):
    """
    Dynamic State Manager.
    
    Manages a bank of states with:
    - Learnable initial states
    - Importance-based pruning
    - State aggregation from hidden states
    
    Args:
        hidden_size: Hidden dimension
        state_dim: State dimension
        num_states: Number of state slots
        num_heads: Heads for state attention
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        num_states: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_states = num_states
        
        # Learnable initial states
        self.initial_states = nn.Parameter(
            torch.randn(1, num_states, state_dim) * 0.02
        )
        
        # Project hidden to state dimension
        self.hidden_proj = nn.Linear(hidden_size, state_dim, bias=False)
        
        # State propagator
        self.propagator = GatedStatePropagator(state_dim)
        
        # Attention for aggregating hidden states
        self.scale = (state_dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.k_proj = nn.Linear(state_dim, state_dim, bias=False)
        self.v_proj = nn.Linear(state_dim, state_dim, bias=False)
    
    def get_initial_states(self, batch_size: int) -> torch.Tensor:
        """Get initial states expanded for batch."""
        return self.initial_states.expand(batch_size, -1, -1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update states based on hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            states: [batch, num_states, state_dim]
            
        Returns:
            Updated states [batch, num_states, state_dim]
        """
        # Project hidden states
        hidden_proj = self.hidden_proj(hidden_states)  # [batch, seq, state_dim]
        
        # Attention: states query hidden states
        q = self.q_proj(states)  # [batch, num_states, state_dim]
        k = self.k_proj(hidden_proj)  # [batch, seq, state_dim]
        v = self.v_proj(hidden_proj)  # [batch, seq, state_dim]
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate
        new_info = torch.matmul(attn, v)  # [batch, num_states, state_dim]
        
        # Update states
        return self.propagator(states, new_info)
