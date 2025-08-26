import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nsm.state_propagator import StatePropagator


def test_state_propagator():
    """Test the StatePropagator with both GRU and LSTM-style gating."""
    
    batch_size = 4
    state_dim = 128
    
    # Test GRU-style gating
    print("Testing GRU-style gating...")
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    updated_state = propagator(prev_state, new_input)
    
    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("GRU-style gating test passed!")
    
    # Test LSTM-style gating
    print("Testing LSTM-style gating...")
    propagator = StatePropagator(state_dim, gate_type='lstm')
    
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    updated_state = propagator(prev_state, new_input)
    
    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("LSTM-style gating test passed!")
    
    # Test differentiability
    print("Testing differentiability...")
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    prev_state = torch.randn(batch_size, state_dim, requires_grad=True)
    new_input = torch.randn(batch_size, state_dim, requires_grad=True)
    
    updated_state = propagator(prev_state, new_input)
    loss = updated_state.sum()
    loss.backward()
    
    assert prev_state.grad is not None, "Gradient not computed for prev_state"
    assert new_input.grad is not None, "Gradient not computed for new_input"
    print("Differentiability test passed!")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_state_propagator()