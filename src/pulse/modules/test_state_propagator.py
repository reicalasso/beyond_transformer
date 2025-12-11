"""
Test suite for the StatePropagator module.
"""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pulse.modules.state_propagator import StatePropagator


def test_single_state_gru():
    """Test GRU-style gating with single state."""
    batch_size = 4
    state_dim = 128

    propagator = StatePropagator(state_dim, gate_type="gru")

    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)

    updated_state = propagator(prev_state, new_input)

    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("✓ GRU-style gating with single state test passed")


def test_single_state_lstm():
    """Test LSTM-style gating with single state."""
    batch_size = 4
    state_dim = 128

    propagator = StatePropagator(state_dim, gate_type="lstm")

    prev_state = torch.randn(batch_size, state_dim)
    new_input = torch.randn(batch_size, state_dim)

    updated_state = propagator(prev_state, new_input)

    assert updated_state.shape == (batch_size, state_dim), "Output shape mismatch"
    print("✓ LSTM-style gating with single state test passed")


def test_single_state_differentiability():
    """Test differentiability with single state."""
    batch_size = 4
    state_dim = 128

    propagator = StatePropagator(state_dim, gate_type="gru")

    prev_state = torch.randn(batch_size, state_dim, requires_grad=True)
    new_input = torch.randn(batch_size, state_dim, requires_grad=True)

    updated_state = propagator(prev_state, new_input)
    loss = updated_state.sum()
    loss.backward()

    assert prev_state.grad is not None, "Gradient not computed for prev_state"
    assert new_input.grad is not None, "Gradient not computed for new_input"
    print("✓ Differentiability with single state test passed")


def test_multi_state_gru():
    """Test GRU-style gating with multiple states."""
    batch_size = 4
    state_dim = 128
    num_states = 8

    propagator = StatePropagator(state_dim, gate_type="gru", enable_communication=True)

    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)

    updated_states = propagator(prev_states, new_inputs)

    assert updated_states.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "Output shape mismatch"
    print("✓ GRU-style gating with multiple states test passed")


def test_multi_state_lstm():
    """Test LSTM-style gating with multiple states."""
    batch_size = 4
    state_dim = 128
    num_states = 8

    propagator = StatePropagator(state_dim, gate_type="lstm", enable_communication=True)

    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)

    updated_states = propagator(prev_states, new_inputs)

    assert updated_states.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "Output shape mismatch"
    print("✓ LSTM-style gating with multiple states test passed")


def test_multi_state_differentiability():
    """Test differentiability with multiple states."""
    batch_size = 4
    state_dim = 128
    num_states = 8

    propagator = StatePropagator(state_dim, gate_type="gru", enable_communication=True)

    prev_states = torch.randn(batch_size, num_states, state_dim, requires_grad=True)
    new_inputs = torch.randn(batch_size, num_states, state_dim, requires_grad=True)

    updated_states = propagator(prev_states, new_inputs)
    loss = updated_states.sum()
    loss.backward()

    assert prev_states.grad is not None, "Gradient not computed for prev_states"
    assert new_inputs.grad is not None, "Gradient not computed for new_inputs"
    print("✓ Differentiability with multiple states test passed")


def test_without_communication():
    """Test without communication."""
    batch_size = 4
    state_dim = 128
    num_states = 8

    propagator = StatePropagator(state_dim, gate_type="gru", enable_communication=False)

    prev_states = torch.randn(batch_size, num_states, state_dim)
    new_inputs = torch.randn(batch_size, num_states, state_dim)

    updated_states = propagator(prev_states, new_inputs)

    assert updated_states.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "Output shape mismatch"
    print("✓ Test without communication passed")


def run_all_tests():
    """Run all tests for the StatePropagator."""
    print("Running StatePropagator tests...")
    print()

    # Single state tests
    print("Single State Tests:")
    test_single_state_gru()
    test_single_state_lstm()
    test_single_state_differentiability()

    print()
    print("Multiple State Tests:")
    test_multi_state_gru()
    test_multi_state_lstm()
    test_multi_state_differentiability()
    test_without_communication()

    print()
    print("✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
