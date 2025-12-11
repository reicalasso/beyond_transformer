"""
Test suite for the SequencePulse model.
"""

import pytest
import torch

from pulse.models.simple_pulse import SequencePulse, SimplePulse


class TestSimplePulseBasic:
    """Basic tests for SimplePulse."""

    def test_initialization(self):
        """Test SimplePulse initialization."""
        model = SimplePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
        )
        
        assert model.input_dim == 64
        assert model.state_dim == 128
        assert model.num_states == 16
        assert model.output_dim == 10

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = SimplePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
        )
        
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output.shape == (4, 10)

    def test_return_states(self):
        """Test forward pass with return_states=True."""
        model = SimplePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
        )
        
        x = torch.randn(4, 64)
        output, states = model(x, return_states=True)
        
        assert output.shape == (4, 10)
        assert states.shape == (4, 16, 128)

    def test_gradient_flow(self):
        """Test gradient flow through SimplePulse."""
        model = SimplePulse(
            input_dim=64,
            state_dim=128,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestSequencePulseBasic:
    """Basic tests for SequencePulse initialization and forward pass."""

    def test_initialization_default(self):
        """Test SequencePulse initialization with default parameters."""
        model = SequencePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
        )
        
        assert model.input_dim == 64
        assert model.state_dim == 128
        assert model.num_states == 16
        assert model.output_dim == 10
        assert model.output_mode == "last"

    def test_initialization_custom_output_mode(self):
        """Test SequencePulse with different output modes."""
        for mode in ["last", "all", "mean"]:
            model = SequencePulse(
                input_dim=32,
                state_dim=64,
                num_states=8,
                output_dim=5,
                output_mode=mode,
            )
            assert model.output_mode == mode

    def test_forward_shape_last(self):
        """Test forward pass output shape with output_mode='last'."""
        batch_size, seq_len = 4, 20
        
        model = SequencePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
            output_mode="last",
        )
        
        x = torch.randn(batch_size, seq_len, 64)
        output = model(x)
        
        assert output.shape == (batch_size, 10)

    def test_forward_shape_all(self):
        """Test forward pass output shape with output_mode='all'."""
        batch_size, seq_len = 4, 20
        
        model = SequencePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
            output_mode="all",
        )
        
        x = torch.randn(batch_size, seq_len, 64)
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 10)

    def test_forward_shape_mean(self):
        """Test forward pass output shape with output_mode='mean'."""
        batch_size, seq_len = 4, 20
        
        model = SequencePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
            output_mode="mean",
        )
        
        x = torch.randn(batch_size, seq_len, 64)
        output = model(x)
        
        assert output.shape == (batch_size, 10)


class TestSequencePulseStates:
    """Tests for SequencePulse state handling."""

    def test_return_all_states(self):
        """Test forward pass with return_all_states=True."""
        batch_size, seq_len = 2, 10
        num_states, state_dim = 8, 64
        
        model = SequencePulse(
            input_dim=32,
            state_dim=state_dim,
            num_states=num_states,
            output_dim=5,
        )
        
        x = torch.randn(batch_size, seq_len, 32)
        output, all_states = model(x, return_all_states=True)
        
        assert all_states.shape == (batch_size, seq_len, num_states, state_dim)

    def test_custom_initial_states(self):
        """Test forward pass with custom initial states."""
        batch_size, seq_len = 2, 10
        num_states, state_dim = 8, 64
        
        model = SequencePulse(
            input_dim=32,
            state_dim=state_dim,
            num_states=num_states,
            output_dim=5,
        )
        
        x = torch.randn(batch_size, seq_len, 32)
        initial_states = torch.zeros(batch_size, num_states, state_dim)
        
        output = model(x, initial_states=initial_states)
        
        assert output.shape == (batch_size, 5)

    def test_get_state_info(self):
        """Test get_state_info method."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
            output_mode="mean",
        )
        
        info = model.get_state_info()
        
        assert info["num_states"] == 8
        assert info["state_dim"] == 64
        assert info["output_mode"] == "mean"


class TestSequencePulseGradients:
    """Tests for gradient flow through SequencePulse."""

    def test_gradient_flow_input(self):
        """Test gradients flow through input."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 10, 32, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_flow_parameters(self):
        """Test parameters receive gradients."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 10, 32)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check initial_states parameter has gradient
        assert model.initial_states.grad is not None

    def test_gradient_flow_all_output_modes(self):
        """Test gradient flow for all output modes."""
        for mode in ["last", "all", "mean"]:
            model = SequencePulse(
                input_dim=32,
                state_dim=64,
                num_states=8,
                output_dim=5,
                output_mode=mode,
            )
            
            x = torch.randn(2, 10, 32, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            
            assert x.grad is not None, f"No gradient for output_mode={mode}"


class TestSequencePulseNumericalStability:
    """Tests for numerical stability of SequencePulse."""

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        model = SequencePulse(
            input_dim=64,
            state_dim=128,
            num_states=16,
            output_dim=10,
        )
        
        x = torch.randn(4, 20, 64)
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_long_sequence(self):
        """Test with long sequences."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 100, 32)  # Long sequence
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert output.shape == (2, 5)

    def test_large_input_values(self):
        """Test stability with large input values."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 10, 32) * 10
        output = model(x)
        
        assert not torch.isnan(output).any()


class TestSequencePulseTraining:
    """Tests for SequencePulse in training scenarios."""

    def test_train_eval_modes(self):
        """Test switching between train and eval modes."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        
        x = torch.randn(2, 10, 32)
        
        model.train()
        train_output = model(x)
        
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        
        assert train_output.shape == eval_output.shape

    def test_deterministic_eval(self):
        """Test that eval mode produces deterministic outputs."""
        model = SequencePulse(
            input_dim=32,
            state_dim=64,
            num_states=8,
            output_dim=5,
        )
        model.eval()
        
        x = torch.randn(2, 10, 32)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
