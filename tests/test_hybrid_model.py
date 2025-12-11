"""
Test suite for the HybridModel modules.
"""

import pytest
import torch

from pulse.models.hybrid_model import AdvancedHybridModel, SequentialHybridModel


class TestAdvancedHybridModelBasic:
    """Basic tests for AdvancedHybridModel initialization and forward pass."""

    def test_initialization_default(self):
        """Test AdvancedHybridModel initialization with default config."""
        model = AdvancedHybridModel()
        
        assert model.input_dim == 784
        assert model.output_dim == 10
        assert model.embedding_dim == 128

    def test_initialization_custom_config(self):
        """Test AdvancedHybridModel initialization with custom config."""
        config = {
            "input_dim": 512,
            "output_dim": 5,
            "embedding_dim": 64,
            "sequence_length": 8,
        }
        model = AdvancedHybridModel(config)
        
        assert model.input_dim == 512
        assert model.output_dim == 5
        assert model.embedding_dim == 64

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        config = {
            "input_dim": 256,
            "output_dim": 10,
            "embedding_dim": 32,
            "sequence_length": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(batch_size, 256)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)

    def test_forward_batch_size_one(self):
        """Test forward pass with batch size of 1."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
            "sequence_length": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(1, 128)
        
        output = model(x)
        
        assert output.shape == (1, 5)


class TestAdvancedHybridModelGradients:
    """Tests for gradient flow through AdvancedHybridModel."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
            "sequence_length": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(2, 128, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameter_gradients(self):
        """Test that parameters receive gradients."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
            "sequence_length": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(2, 128)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "No parameters received gradients"


class TestAdvancedHybridModelMethods:
    """Tests for AdvancedHybridModel methods."""

    def test_get_attention_weights(self):
        """Test get_attention_weights method."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
            "sequence_length": 4,
            "attention_heads": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(2, 128)
        
        attn_weights = model.get_attention_weights(x)
        
        # Shape should be [batch_size, num_heads, seq_len, seq_len]
        assert attn_weights.dim() == 4
        assert attn_weights.shape[0] == 2  # batch_size

    def test_get_model_info(self):
        """Test get_model_info method."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
        }
        
        model = AdvancedHybridModel(config)
        info = model.get_model_info()
        
        assert "config" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert "parameter_details" in info
        assert info["total_parameters"] > 0


class TestSequentialHybridModelBasic:
    """Basic tests for SequentialHybridModel."""

    def test_initialization_default(self):
        """Test SequentialHybridModel initialization with defaults."""
        model = SequentialHybridModel()
        
        assert model.input_dim == 784
        assert model.output_dim == 10

    def test_initialization_custom(self):
        """Test SequentialHybridModel initialization with custom params."""
        model = SequentialHybridModel(input_dim=256, output_dim=5)
        
        assert model.input_dim == 256
        assert model.output_dim == 5

    def test_forward_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        
        model = SequentialHybridModel(input_dim=256, output_dim=10)
        x = torch.randn(batch_size, 256)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)

    def test_forward_batch_size_one(self):
        """Test forward pass with batch size of 1."""
        model = SequentialHybridModel(input_dim=128, output_dim=5)
        x = torch.randn(1, 128)
        
        output = model(x)
        
        assert output.shape == (1, 5)


class TestSequentialHybridModelGradients:
    """Tests for gradient flow through SequentialHybridModel."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = SequentialHybridModel(input_dim=128, output_dim=5)
        x = torch.randn(2, 128, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_differentiability(self):
        """Test full differentiability of the model."""
        model = SequentialHybridModel(input_dim=64, output_dim=3)
        x = torch.randn(2, 64, requires_grad=True)
        
        # Forward pass
        output = model(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"


class TestHybridModelNumericalStability:
    """Tests for numerical stability of hybrid models."""

    def test_advanced_no_nan_output(self):
        """Test AdvancedHybridModel produces no NaN values."""
        config = {
            "input_dim": 128,
            "output_dim": 5,
            "embedding_dim": 32,
            "sequence_length": 4,
        }
        
        model = AdvancedHybridModel(config)
        x = torch.randn(4, 128)
        
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sequential_no_nan_output(self):
        """Test SequentialHybridModel produces no NaN values."""
        model = SequentialHybridModel(input_dim=128, output_dim=5)
        x = torch.randn(4, 128)
        
        output = model(x)
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_large_input_values(self):
        """Test stability with large input values."""
        model = SequentialHybridModel(input_dim=64, output_dim=3)
        x = torch.randn(2, 64) * 10
        
        output = model(x)
        
        assert not torch.isnan(output).any()


class TestHybridModelTraining:
    """Tests for hybrid models in training scenarios."""

    def test_train_eval_modes(self):
        """Test switching between train and eval modes."""
        model = SequentialHybridModel(input_dim=64, output_dim=3)
        x = torch.randn(2, 64)
        
        # Train mode
        model.train()
        train_output = model(x)
        
        # Eval mode
        model.eval()
        with torch.no_grad():
            eval_output = model(x)
        
        assert train_output.shape == eval_output.shape

    def test_multiple_forward_passes(self):
        """Test multiple forward passes don't accumulate state incorrectly."""
        model = SequentialHybridModel(input_dim=64, output_dim=3)
        x = torch.randn(2, 64)
        
        # Multiple forward passes
        output1 = model(x)
        output2 = model(x)
        output3 = model(x)
        
        # Outputs should be identical for same input (in eval mode)
        model.eval()
        with torch.no_grad():
            eval_out1 = model(x)
            eval_out2 = model(x)
        
        assert torch.allclose(eval_out1, eval_out2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
