"""
Unit tests for the StatePropagator module.
"""

import numpy as np
import pytest
import torch

from pulse import StatePropagator


class TestStatePropagator:
    """Test suite for StatePropagator class."""

    @pytest.fixture
    def propagator_gru(self, sample_state_dim):
        """Create a GRU-based StatePropagator for testing."""
        return StatePropagator(
            state_dim=sample_state_dim,
            gate_type="gru",
            num_heads=4,
            enable_communication=True,
        )

    @pytest.fixture
    def propagator_lstm(self, sample_state_dim):
        """Create an LSTM-based StatePropagator for testing."""
        return StatePropagator(
            state_dim=sample_state_dim,
            gate_type="lstm",
            num_heads=4,
            enable_communication=True,
        )

    @pytest.fixture
    def propagator_no_comm(self, sample_state_dim):
        """Create a StatePropagator without communication for testing."""
        return StatePropagator(
            state_dim=sample_state_dim,
            gate_type="gru",
            num_heads=4,
            enable_communication=False,
        )

    def test_initialization(self, sample_state_dim):
        """Test StatePropagator initialization."""
        # Test GRU initialization
        propagator = StatePropagator(
            state_dim=sample_state_dim,
            gate_type="gru",
            num_heads=4,
            enable_communication=True,
        )

        assert propagator.state_dim == sample_state_dim
        assert propagator.gate_type == "gru"
        assert propagator.enable_communication is True
        assert hasattr(propagator, "reset_gate")
        assert hasattr(propagator, "update_gate")
        assert hasattr(propagator, "candidate_state")
        assert hasattr(propagator, "attention")

    def test_initialization_lstm(self, sample_state_dim):
        """Test LSTM-based StatePropagator initialization."""
        propagator = StatePropagator(
            state_dim=sample_state_dim,
            gate_type="lstm",
            num_heads=4,
            enable_communication=True,
        )

        assert propagator.gate_type == "lstm"
        assert hasattr(propagator, "forget_gate")
        assert hasattr(propagator, "input_gate")
        assert hasattr(propagator, "output_gate")
        assert hasattr(propagator, "candidate_state")

    def test_invalid_gate_type(self, sample_state_dim):
        """Test initialization with invalid gate type."""
        with pytest.raises(ValueError):
            StatePropagator(
                state_dim=sample_state_dim,
                gate_type="invalid",
                num_heads=4,
                enable_communication=True,
            )

    def test_single_state_update_gru(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test single state update with GRU."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        updated_state = propagator_gru._single_state_update(prev_state, new_input)

        assert updated_state.shape == prev_state.shape
        assert torch.isfinite(updated_state).all(), "Updated state contains NaN or Inf"

        # Check that state has changed
        assert not torch.equal(prev_state, updated_state), "State should have changed"

    def test_single_state_update_lstm(
        self, propagator_lstm, sample_batch_size, sample_state_dim
    ):
        """Test single state update with LSTM."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        updated_state = propagator_lstm._single_state_update(prev_state, new_input)

        assert updated_state.shape == prev_state.shape
        assert torch.isfinite(updated_state).all(), "Updated state contains NaN or Inf"

        # Check that state has changed
        assert not torch.equal(prev_state, updated_state), "State should have changed"

    def test_multi_state_update_gru(
        self, propagator_gru, sample_batch_size, sample_num_states, sample_state_dim
    ):
        """Test multi-state update with GRU."""
        prev_states = torch.randn(
            sample_batch_size, sample_num_states, sample_state_dim
        )
        new_inputs = torch.randn(sample_batch_size, sample_num_states, sample_state_dim)

        updated_states = propagator_gru._multi_state_update(prev_states, new_inputs)

        assert updated_states.shape == prev_states.shape
        assert torch.isfinite(updated_states).all(), "Updated states contain NaN or Inf"

        # Check that states have changed
        assert not torch.equal(
            prev_states, updated_states
        ), "States should have changed"

    def test_multi_state_update_lstm(
        self, propagator_lstm, sample_batch_size, sample_num_states, sample_state_dim
    ):
        """Test multi-state update with LSTM."""
        prev_states = torch.randn(
            sample_batch_size, sample_num_states, sample_state_dim
        )
        new_inputs = torch.randn(sample_batch_size, sample_num_states, sample_state_dim)

        updated_states = propagator_lstm._multi_state_update(prev_states, new_inputs)

        assert updated_states.shape == prev_states.shape
        assert torch.isfinite(updated_states).all(), "Updated states contain NaN or Inf"

        # Check that states have changed
        assert not torch.equal(
            prev_states, updated_states
        ), "States should have changed"

    def test_forward_single_state(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test forward pass with single state."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        updated_state = propagator_gru(prev_state, new_input)

        assert updated_state.shape == prev_state.shape
        assert torch.isfinite(updated_state).all(), "Updated state contains NaN or Inf"

    def test_forward_multi_state(
        self, propagator_gru, sample_batch_size, sample_num_states, sample_state_dim
    ):
        """Test forward pass with multiple states."""
        prev_states = torch.randn(
            sample_batch_size, sample_num_states, sample_state_dim
        )
        new_inputs = torch.randn(sample_batch_size, sample_num_states, sample_state_dim)

        updated_states = propagator_gru(prev_states, new_inputs)

        assert updated_states.shape == prev_states.shape
        assert torch.isfinite(updated_states).all(), "Updated states contain NaN or Inf"

    def test_forward_no_communication(
        self, propagator_no_comm, sample_batch_size, sample_num_states, sample_state_dim
    ):
        """Test forward pass without communication."""
        prev_states = torch.randn(
            sample_batch_size, sample_num_states, sample_state_dim
        )
        new_inputs = torch.randn(sample_batch_size, sample_num_states, sample_state_dim)

        updated_states = propagator_no_comm(prev_states, new_inputs)

        assert updated_states.shape == prev_states.shape
        assert torch.isfinite(updated_states).all(), "Updated states contain NaN or Inf"

    def test_gru_step_properties(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test properties of GRU step."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        # Test with zero inputs
        zero_state = torch.zeros(sample_batch_size, sample_state_dim)
        zero_input = torch.zeros(sample_batch_size, sample_state_dim)

        updated_zero = propagator_gru._single_state_update(zero_state, zero_input)
        assert torch.isfinite(updated_zero).all(), "Zero input produces invalid output"

    def test_lstm_step_properties(
        self, propagator_lstm, sample_batch_size, sample_state_dim
    ):
        """Test properties of LSTM step."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        # Test with zero inputs
        zero_state = torch.zeros(sample_batch_size, sample_state_dim)
        zero_input = torch.zeros(sample_batch_size, sample_state_dim)

        updated_zero = propagator_lstm._single_state_update(zero_state, zero_input)
        assert torch.isfinite(updated_zero).all(), "Zero input produces invalid output"

    def test_state_preservation_bounds(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test that state values stay within reasonable bounds."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        updated_state = propagator_gru._single_state_update(prev_state, new_input)

        # Check that values don't explode
        assert torch.max(torch.abs(updated_state)) < 100, "State values exploded"

        # Check that gradients flow
        prev_state.requires_grad_(True)
        new_input.requires_grad_(True)

        updated_state = propagator_gru._single_state_update(prev_state, new_input)
        loss = updated_state.sum()
        loss.backward()

        assert prev_state.grad is not None, "Gradients should flow through prev_state"
        assert new_input.grad is not None, "Gradients should flow through new_input"
        assert torch.isfinite(prev_state.grad).all(), "Gradients contain NaN or Inf"

    def test_consistency_across_runs(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test that the same input produces the same output."""
        prev_state = torch.randn(sample_batch_size, sample_state_dim)
        new_input = torch.randn(sample_batch_size, sample_state_dim)

        # Run twice with same inputs
        output1 = propagator_gru._single_state_update(
            prev_state.clone(), new_input.clone()
        )
        output2 = propagator_gru._single_state_update(
            prev_state.clone(), new_input.clone()
        )

        # Should be identical (within floating point precision)
        assert torch.allclose(
            output1, output2, atol=1e-6
        ), "Outputs should be consistent"

    def test_differentiability(
        self, propagator_gru, sample_batch_size, sample_state_dim
    ):
        """Test that the propagator is fully differentiable."""
        prev_state = torch.randn(
            sample_batch_size, sample_state_dim, requires_grad=True
        )
        new_input = torch.randn(sample_batch_size, sample_state_dim, requires_grad=True)

        updated_state = propagator_gru._single_state_update(prev_state, new_input)

        # Compute loss and backpropagate
        loss = updated_state.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert prev_state.grad is not None
        assert new_input.grad is not None
        assert torch.isfinite(prev_state.grad).all()
        assert torch.isfinite(new_input.grad).all()

    @pytest.mark.parametrize("gate_type", ["gru", "lstm"])
    def test_gate_types(self, gate_type, sample_state_dim):
        """Test both gate types work correctly."""
        propagator = StatePropagator(
            state_dim=sample_state_dim,
            gate_type=gate_type,
            num_heads=2,
            enable_communication=True,
        )

        batch_size = 2
        num_states = 4

        prev_states = torch.randn(batch_size, num_states, sample_state_dim)
        new_inputs = torch.randn(batch_size, num_states, sample_state_dim)

        updated_states = propagator(prev_states, new_inputs)

        assert updated_states.shape == prev_states.shape
        assert torch.isfinite(updated_states).all()
