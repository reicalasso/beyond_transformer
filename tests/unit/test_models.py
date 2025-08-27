"""
Unit tests for NSM models.
"""

import torch
import pytest
import numpy as np
from nsm.models.simple_nsm import SimpleNSM
from nsm.models.hybrid_model import AdvancedHybridModel, SequentialHybridModel


class TestSimpleNSM:
    """Test suite for SimpleNSM model."""

    @pytest.fixture
    def simple_nsm(self):
        """Create a SimpleNSM for testing."""
        return SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )

    def test_initialization(self):
        """Test SimpleNSM initialization."""
        model = SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )
        
        assert model.input_dim == 64
        assert model.state_dim == 32
        assert model.num_states == 8
        assert model.output_dim == 10
        assert model.gate_type == 'gru'
        
        # Check that components are initialized
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'state_propagator')
        assert hasattr(model, 'output_projection')
        assert hasattr(model, 'initial_states')

    def test_forward_pass(self, simple_nsm, sample_batch_size):
        """Test forward pass of SimpleNSM."""
        x = torch.randn(sample_batch_size, 64)
        
        output = simple_nsm(x)
        
        assert output.shape == (sample_batch_size, 10)
        assert torch.isfinite(output).all()

    def test_different_input_sizes(self):
        """Test SimpleNSM with different input sizes."""
        # Test with different input dimensions
        for input_dim in [32, 64, 128]:
            model = SimpleNSM(
                input_dim=input_dim,
                state_dim=32,
                num_states=8,
                output_dim=5,
                gate_type='gru'
            )
            
            batch_size = 4
            x = torch.randn(batch_size, input_dim)
            output = model(x)
            
            assert output.shape == (batch_size, 5)
            assert torch.isfinite(output).all()

    def test_different_gate_types(self):
        """Test SimpleNSM with different gate types."""
        for gate_type in ['gru', 'lstm']:
            model = SimpleNSM(
                input_dim=64,
                state_dim=32,
                num_states=8,
                output_dim=10,
                gate_type=gate_type
            )
            
            batch_size = 4
            x = torch.randn(batch_size, 64)
            output = model(x)
            
            assert output.shape == (batch_size, 10)
            assert torch.isfinite(output).all()

    def test_parameter_count(self, simple_nsm):
        """Test that model has reasonable parameter count."""
        total_params = sum(p.numel() for p in simple_nsm.parameters())
        trainable_params = sum(p.numel() for p in simple_nsm.parameters() if p.requires_grad)
        
        # Should have parameters
        assert total_params > 0
        assert trainable_params > 0
        assert total_params == trainable_params  # All should be trainable

    def test_backward_pass(self, simple_nsm, sample_batch_size):
        """Test backward pass of SimpleNSM."""
        x = torch.randn(sample_batch_size, 64)
        
        # Forward pass
        output = simple_nsm(x)
        
        # Compute loss and backward pass
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        for param in simple_nsm.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    def test_state_initialization(self, simple_nsm):
        """Test state initialization."""
        initial_states = simple_nsm.initial_states
        
        assert initial_states.shape == (1, simple_nsm.num_states, simple_nsm.state_dim)
        assert torch.isfinite(initial_states).all()

    def test_model_device_movement(self, simple_nsm):
        """Test moving model between devices."""
        # Test CPU (always available)
        simple_nsm.cpu()
        assert next(simple_nsm.parameters()).device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            simple_nsm.cuda()
            assert next(simple_nsm.parameters()).device.type == 'cuda'

    def test_batch_independence(self, sample_batch_size):
        """Test that batches are processed independently."""
        model = SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )
        
        # Create two identical batches
        x1 = torch.randn(sample_batch_size, 64)
        x2 = x1.clone()
        
        # Process together
        x_combined = torch.cat([x1, x2], dim=0)
        output_combined = model(x_combined)
        
        # Process separately
        output1 = model(x1)
        output2 = model(x2)
        output_separate = torch.cat([output1, output2], dim=0)
        
        # Should be identical (within floating point precision)
        assert torch.allclose(output_combined, output_separate, atol=1e-6)


class TestAdvancedHybridModel:
    """Test suite for AdvancedHybridModel."""

    def test_initialization(self):
        """Test AdvancedHybridModel initialization."""
        config = {
            'input_dim': 64,
            'output_dim': 10,
            'embedding_dim': 32,
            'sequence_length': 8,
            'ssm_dim': 32,
            'ntm_mem_size': 32,
            'ntm_mem_dim': 16,
            'rnn_hidden_dim': 32,
            'attention_heads': 4
        }
        
        model = AdvancedHybridModel(config)
        
        # Check config is stored
        assert hasattr(model, 'config')
        
        # Check components are initialized
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'ssm_block')
        assert hasattr(model, 'ntm_memory')
        assert hasattr(model, 'rnn_memory')
        assert hasattr(model, 'output_projection')

    def test_forward_pass(self):
        """Test forward pass of AdvancedHybridModel."""
        config = {
            'input_dim': 64,
            'output_dim': 10,
            'embedding_dim': 32,
            'sequence_length': 8,
            'ssm_dim': 32,
            'ntm_mem_size': 32,
            'ntm_mem_dim': 16,
            'rnn_hidden_dim': 32,
            'attention_heads': 4
        }
        
        model = AdvancedHybridModel(config)
        batch_size = 4
        x = torch.randn(batch_size, 64)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert torch.isfinite(output).all()

    def test_get_model_info(self):
        """Test getting model information."""
        config = {
            'input_dim': 64,
            'output_dim': 10,
            'embedding_dim': 32,
            'sequence_length': 8,
            'ssm_dim': 32,
            'ntm_mem_size': 32,
            'ntm_mem_dim': 16,
            'rnn_hidden_dim': 32,
            'attention_heads': 4
        }
        
        model = AdvancedHybridModel(config)
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'config' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info


class TestSequentialHybridModel:
    """Test suite for SequentialHybridModel."""

    def test_initialization(self):
        """Test SequentialHybridModel initialization."""
        model = SequentialHybridModel(input_dim=64, output_dim=10)
        
        assert model.input_dim == 64
        assert model.output_dim == 10
        
        # Check components are initialized
        assert hasattr(model, 'input_to_sequence')
        assert hasattr(model, 'positional_encoding')
        assert hasattr(model, 'attention')
        assert hasattr(model, 'ssm')
        assert hasattr(model, 'ntm')
        assert hasattr(model, 'rnn')
        assert hasattr(model, 'output_layer')

    def test_forward_pass(self):
        """Test forward pass of SequentialHybridModel."""
        model = SequentialHybridModel(input_dim=64, output_dim=10)
        batch_size = 4
        x = torch.randn(batch_size, 64)
        
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        assert torch.isfinite(output).all()

    def test_different_architectures(self):
        """Test SequentialHybridModel with different dimensions."""
        for input_dim, output_dim in [(32, 5), (64, 10), (128, 20)]:
            model = SequentialHybridModel(input_dim=input_dim, output_dim=output_dim)
            batch_size = 2
            x = torch.randn(batch_size, input_dim)
            output = model(x)
            
            assert output.shape == (batch_size, output_dim)
            assert torch.isfinite(output).all()


# Integration tests
class TestModelIntegration:
    """Integration tests for NSM models."""

    @pytest.mark.integration
    def test_training_loop_basic(self):
        """Test basic training loop with SimpleNSM."""
        model = SimpleNSM(
            input_dim=32,
            state_dim=16,
            num_states=4,
            output_dim=3,
            gate_type='gru'
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 32)
        y_true = torch.randint(0, 3, (batch_size,))
        
        # Forward pass
        output = model(x)
        assert output.shape == (batch_size, 3)
        
        # Loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, y_true)
        assert torch.isfinite(loss).item()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for param in model.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()

    @pytest.mark.integration
    def test_model_comparison(self):
        """Compare different NSM models."""
        # Create models
        simple_model = SimpleNSM(
            input_dim=32,
            state_dim=16,
            num_states=4,
            output_dim=3,
            gate_type='gru'
        )
        
        sequential_model = SequentialHybridModel(input_dim=32, output_dim=3)
        
        # Same input
        batch_size = 4
        x = torch.randn(batch_size, 32)
        
        # Forward passes
        simple_output = simple_model(x)
        sequential_output = sequential_model(x)
        
        # Both should produce valid outputs
        assert simple_output.shape == (batch_size, 3)
        assert sequential_output.shape == (batch_size, 3)
        assert torch.isfinite(simple_output).all()
        assert torch.isfinite(sequential_output).all()

    @pytest.mark.slow
    def test_parameter_scaling(self):
        """Test that parameter count scales reasonably."""
        small_model = SimpleNSM(
            input_dim=32,
            state_dim=16,
            num_states=4,
            output_dim=10,
            gate_type='gru'
        )
        
        large_model = SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )
        
        small_params = sum(p.numel() for p in small_model.parameters())
        large_params = sum(p.numel() for p in large_model.parameters())
        
        # Larger model should have more parameters
        assert large_params > small_params
        
        # But not exponentially more (reasonable scaling)
        ratio = large_params / small_params
        assert 2 < ratio < 10  # Reasonable scaling


# Performance tests
class TestModelPerformance:
    """Performance tests for NSM models."""

    @pytest.mark.slow
    def test_forward_pass_speed(self):
        """Test forward pass performance."""
        model = SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )
        
        batch_size = 32
        x = torch.randn(batch_size, 64)
        
        # Warm up
        _ = model(x)
        
        # Time multiple forward passes
        import time
        start_time = time.time()
        
        num_iterations = 100
        for _ in range(num_iterations):
            _ = model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        # Should be reasonably fast (less than 100ms per forward pass)
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.4f}s"

    @pytest.mark.gpu
    def test_gpu_memory_efficiency(self):
        """Test GPU memory efficiency."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleNSM(
            input_dim=64,
            state_dim=32,
            num_states=8,
            output_dim=10,
            gate_type='gru'
        )
        
        model.cuda()
        
        batch_size = 16
        x = torch.randn(batch_size, 64).cuda()
        
        # Clear cache and get initial memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        output = model(x)
        torch.cuda.synchronize()
        peak_memory = torch.cuda.memory_allocated()
        
        # Memory usage should be reasonable
        memory_used = peak_memory - initial_memory
        assert memory_used < 100 * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.2f} MB"