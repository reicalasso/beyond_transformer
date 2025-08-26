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
    
    # Test GRU-style gating with single state
    print("Testing GRU-style gating with single state...")
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    updated_state = propagator(prev_state, new_input)
    
    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("GRU-style gating test passed!")
    
    # Test LSTM-style gating with single state
    print("Testing LSTM-style gating with single state...")
    propagator = StatePropagator(state_dim, gate_type='lstm')
    
    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)
    
    updated_state = propagator(prev_state, new_input)
    
    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("LSTM-style gating test passed!")
    
    # Test differentiability with single state
    print("Testing differentiability with single state...")
    propagator = StatePropagator(state_dim, gate_type='gru')
    
    prev_state = torch.randn(batch_size, state_dim, requires_grad=True)
    new_input = torch.randn(batch_size, state_dim, requires_grad=True)
    
    updated_state = propagator(prev_state, new_input)
    loss = updated_state.sum()
    loss.backward()
    
    assert prev_state.grad is not None, "Gradient not computed for prev_state"
    assert new_input.grad is not None, "Gradient not computed for new_input"
    print("Differentiability test passed!")
    
    print("\n--- Testing multiple states with communication ---")
    
    # Test with multiple states and communication
    num_states = 8
    
    # Test GRU-style gating with multiple states
    print("Testing GRU-style gating with multiple states...")
    propagator_multi = StatePropagator(state_dim, gate_type='gru', enable_communication=True)
    
    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)
    
    updated_states = propagator_multi(prev_states, new_inputs)
    
    assert updated_states.shape == (batch_size, num_states, state_dim), "Output shape mismatch"
    print("GRU-style gating with multiple states test passed!")
    
    # Test LSTM-style gating with multiple states
    print("Testing LSTM-style gating with multiple states...")
    propagator_multi = StatePropagator(state_dim, gate_type='lstm', enable_communication=True)
    
    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)
    
    updated_states = propagator_multi(prev_states, new_inputs)
    
    assert updated_states.shape == (batch_size, num_states, state_dim), "Output shape mismatch"
    print("LSTM-style gating with multiple states test passed!")
    
    # Test differentiability with multiple states
    print("Testing differentiability with multiple states...")
    propagator_multi = StatePropagator(state_dim, gate_type='gru', enable_communication=True)
    
    prev_states = torch.randn(batch_size, num_states, state_dim, requires_grad=True)
    new_inputs = torch.randn(batch_size, num_states, state_dim, requires_grad=True)
    
    updated_states = propagator_multi(prev_states, new_inputs)
    loss = updated_states.sum()
    loss.backward()
    
    assert prev_states.grad is not None, "Gradient not computed for prev_states"
    assert new_inputs.grad is not None, "Gradient not computed for new_inputs"
    print("Differentiability test with multiple states passed!")
    
    # Test without communication
    print("Testing without communication...")
    propagator_no_comm = StatePropagator(state_dim, gate_type='gru', enable_communication=False)
    
    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)
    
    updated_states = propagator_no_comm(prev_states, new_inputs)
    
    assert updated_states.shape == (batch_size, num_states, state_dim), "Output shape mismatch"
    print("Test without communication passed!")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_state_propagator()