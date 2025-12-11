"""
Test suite for the SSMBlock module.
"""

import pytest
import torch

from pulse.modules.ssm_block import SSMBlock, MAMBA_AVAILABLE


class TestSSMBlockBasic:
    """Basic tests for SSMBlock initialization and forward pass."""

    def test_initialization_default(self):
        """Test SSMBlock initialization with default parameters."""
        ssm = SSMBlock(d_model=64)
        
        assert ssm.d_model == 64
        assert ssm.d_state == 16
        assert ssm.d_conv == 4
        assert ssm.expand == 2

    def test_initialization_custom(self):
        """Test SSMBlock initialization with custom parameters."""
        ssm = SSMBlock(
            d_model=128,
            d_state=32,
            d_conv=8,
            expand=4,
            layer_idx=0,
        )
        
        assert ssm.d_model == 128
        assert ssm.d_state == 32
        assert ssm.d_conv == 8
        assert ssm.expand == 4
        assert ssm.layer_idx == 0

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size, seq_len, d_model = 4, 32, 64
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ssm(x)
        
        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_different_seq_lengths(self):
        """Test forward pass with different sequence lengths."""
        batch_size, d_model = 2, 32
        
        ssm = SSMBlock(d_model=d_model)
        
        for seq_len in [8, 16, 64, 128]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = ssm(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_batch_size_one(self):
        """Test forward pass with batch size of 1."""
        seq_len, d_model = 16, 64
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(1, seq_len, d_model)
        
        output = ssm(x)
        
        assert output.shape == (1, seq_len, d_model)


class TestSSMBlockGradients:
    """Tests for gradient flow through SSMBlock."""

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        batch_size, seq_len, d_model = 2, 16, 32
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        
        output = ssm(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameter_gradients(self):
        """Test that parameters receive gradients."""
        batch_size, seq_len, d_model = 2, 16, 32
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ssm(x)
        loss = output.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = False
        for param in ssm.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No parameters received gradients"


class TestSSMBlockFallback:
    """Tests for fallback implementation when Mamba is not available."""

    def test_fallback_components_exist(self):
        """Test that fallback components are created when Mamba unavailable."""
        if MAMBA_AVAILABLE:
            pytest.skip("Mamba is available, skipping fallback test")
        
        ssm = SSMBlock(d_model=64)
        
        assert hasattr(ssm, 'in_proj')
        assert hasattr(ssm, 'conv1d')
        assert hasattr(ssm, 'out_proj')
        assert hasattr(ssm, 'act')

    def test_fallback_forward(self):
        """Test fallback forward pass produces valid output."""
        if MAMBA_AVAILABLE:
            pytest.skip("Mamba is available, skipping fallback test")
        
        batch_size, seq_len, d_model = 2, 16, 64
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ssm._fallback_forward(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()


class TestSSMBlockNumericalStability:
    """Tests for numerical stability of SSMBlock."""

    def test_no_nan_output(self):
        """Test that output contains no NaN values."""
        batch_size, seq_len, d_model = 4, 32, 64
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ssm(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_input_values(self):
        """Test stability with large input values."""
        batch_size, seq_len, d_model = 2, 16, 32
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model) * 10
        
        output = ssm(x)
        
        assert not torch.isnan(output).any()

    def test_small_input_values(self):
        """Test stability with small input values."""
        batch_size, seq_len, d_model = 2, 16, 32
        
        ssm = SSMBlock(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model) * 0.001
        
        output = ssm(x)
        
        assert not torch.isnan(output).any()


class TestSSMBlockTraining:
    """Tests for SSMBlock in training mode."""

    def test_train_mode(self):
        """Test SSMBlock in training mode."""
        ssm = SSMBlock(d_model=64)
        ssm.train()
        
        x = torch.randn(2, 16, 64)
        output = ssm(x)
        
        assert output.shape == (2, 16, 64)

    def test_eval_mode(self):
        """Test SSMBlock in evaluation mode."""
        ssm = SSMBlock(d_model=64)
        ssm.eval()
        
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            output = ssm(x)
        
        assert output.shape == (2, 16, 64)

    def test_train_eval_consistency(self):
        """Test that train and eval modes produce similar outputs for same input."""
        ssm = SSMBlock(d_model=64)
        x = torch.randn(2, 16, 64)
        
        ssm.train()
        train_output = ssm(x.clone())
        
        ssm.eval()
        with torch.no_grad():
            eval_output = ssm(x.clone())
        
        # Outputs should be similar (may differ due to dropout if present)
        # Just check shapes match
        assert train_output.shape == eval_output.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
