"""
Test suite for the NTMMemory module.
"""

import pytest
import torch

from pulse.modules.ntm_memory import NTMMemory


class TestNTMMemoryBasic:
    """Basic tests for NTMMemory initialization and forward pass."""

    def test_initialization(self):
        """Test NTMMemory initialization with default parameters."""
        ntm = NTMMemory(mem_size=64, mem_dim=16)
        
        assert ntm.mem_size == 64
        assert ntm.mem_dim == 16
        assert ntm.num_read_heads == 1
        assert ntm.num_write_heads == 1
        assert ntm.memory.shape == (64, 16)

    def test_initialization_custom_heads(self):
        """Test NTMMemory initialization with custom head counts."""
        ntm = NTMMemory(
            mem_size=128,
            mem_dim=20,
            num_read_heads=2,
            num_write_heads=2,
        )
        
        assert ntm.num_read_heads == 2
        assert ntm.num_write_heads == 2
        assert ntm.read_weights.shape == (2, 128)
        assert ntm.write_weights.shape == (2, 128)

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        batch_size = 4
        mem_size, mem_dim = 64, 16
        num_read_heads, num_write_heads = 1, 1
        
        ntm = NTMMemory(
            mem_size=mem_size,
            mem_dim=mem_dim,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
        )
        
        # Create inputs
        read_keys = torch.randn(batch_size, num_read_heads, mem_dim)
        write_keys = torch.randn(batch_size, num_write_heads, mem_dim)
        read_strengths = torch.randn(batch_size, num_read_heads)
        write_strengths = torch.randn(batch_size, num_write_heads)
        erase_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
        add_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
        
        # Forward pass
        read_vectors, memory_state = ntm(
            read_keys,
            write_keys,
            read_strengths,
            write_strengths,
            erase_vectors,
            add_vectors,
        )
        
        assert read_vectors.shape == (batch_size, num_read_heads, mem_dim)
        assert memory_state.shape == (mem_size, mem_dim)

    def test_forward_multiple_heads(self):
        """Test forward pass with multiple read/write heads."""
        batch_size = 2
        mem_size, mem_dim = 32, 8
        num_read_heads, num_write_heads = 3, 2
        
        ntm = NTMMemory(
            mem_size=mem_size,
            mem_dim=mem_dim,
            num_read_heads=num_read_heads,
            num_write_heads=num_write_heads,
        )
        
        read_keys = torch.randn(batch_size, num_read_heads, mem_dim)
        write_keys = torch.randn(batch_size, num_write_heads, mem_dim)
        read_strengths = torch.randn(batch_size, num_read_heads)
        write_strengths = torch.randn(batch_size, num_write_heads)
        erase_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
        add_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
        
        read_vectors, _ = ntm(
            read_keys,
            write_keys,
            read_strengths,
            write_strengths,
            erase_vectors,
            add_vectors,
        )
        
        assert read_vectors.shape == (batch_size, num_read_heads, mem_dim)


class TestNTMMemoryOperations:
    """Tests for NTMMemory read/write operations."""

    def test_reset_memory(self):
        """Test memory reset functionality."""
        ntm = NTMMemory(mem_size=32, mem_dim=8)
        
        # Modify memory
        ntm.memory.fill_(1.0)
        
        # Reset
        ntm.reset_memory()
        
        assert torch.allclose(ntm.memory, torch.zeros(32, 8))
        assert torch.allclose(ntm.read_weights, torch.ones(1, 32) / 32)

    def test_get_memory_state(self):
        """Test get_memory_state returns a copy."""
        ntm = NTMMemory(mem_size=16, mem_dim=4)
        
        state = ntm.get_memory_state()
        
        # Modify the returned state
        state.fill_(999)
        
        # Original should be unchanged
        assert not torch.allclose(ntm.memory, state)

    def test_get_read_weights(self):
        """Test get_read_weights returns correct shape."""
        ntm = NTMMemory(mem_size=32, mem_dim=8, num_read_heads=2)
        
        weights = ntm.get_read_weights()
        
        assert weights.shape == (2, 32)

    def test_get_write_weights(self):
        """Test get_write_weights returns correct shape."""
        ntm = NTMMemory(mem_size=32, mem_dim=8, num_write_heads=3)
        
        weights = ntm.get_write_weights()
        
        assert weights.shape == (3, 32)


class TestNTMMemoryGradients:
    """Tests for gradient flow through NTMMemory."""

    def test_gradient_flow_read_keys(self):
        """Test gradients flow through read keys."""
        batch_size = 2
        mem_size, mem_dim = 16, 8
        
        ntm = NTMMemory(mem_size=mem_size, mem_dim=mem_dim)
        
        read_keys = torch.randn(batch_size, 1, mem_dim, requires_grad=True)
        write_keys = torch.randn(batch_size, 1, mem_dim)
        read_strengths = torch.randn(batch_size, 1)
        write_strengths = torch.randn(batch_size, 1)
        erase_vectors = torch.randn(batch_size, 1, mem_dim)
        add_vectors = torch.randn(batch_size, 1, mem_dim)
        
        read_vectors, _ = ntm(
            read_keys,
            write_keys,
            read_strengths,
            write_strengths,
            erase_vectors,
            add_vectors,
        )
        
        loss = read_vectors.sum()
        loss.backward()
        
        assert read_keys.grad is not None
        assert not torch.isnan(read_keys.grad).any()

    def test_gradient_flow_write_vectors(self):
        """Test gradients flow through write vectors."""
        batch_size = 2
        mem_size, mem_dim = 16, 8
        
        ntm = NTMMemory(mem_size=mem_size, mem_dim=mem_dim)
        
        read_keys = torch.randn(batch_size, 1, mem_dim)
        write_keys = torch.randn(batch_size, 1, mem_dim, requires_grad=True)
        read_strengths = torch.randn(batch_size, 1)
        write_strengths = torch.randn(batch_size, 1)
        erase_vectors = torch.randn(batch_size, 1, mem_dim, requires_grad=True)
        add_vectors = torch.randn(batch_size, 1, mem_dim, requires_grad=True)
        
        read_vectors, _ = ntm(
            read_keys,
            write_keys,
            read_strengths,
            write_strengths,
            erase_vectors,
            add_vectors,
        )
        
        loss = read_vectors.sum()
        loss.backward()
        
        assert write_keys.grad is not None
        assert erase_vectors.grad is not None
        assert add_vectors.grad is not None


class TestNTMMemoryContentAddressing:
    """Tests for content-based addressing mechanism."""

    def test_content_addressing_similar_keys(self):
        """Test that similar keys produce similar attention weights."""
        batch_size = 2
        mem_size, mem_dim = 16, 8
        
        ntm = NTMMemory(mem_size=mem_size, mem_dim=mem_dim)
        
        # Initialize memory with distinct patterns
        ntm.memory.data = torch.randn(mem_size, mem_dim)
        
        # Use the first memory slot as key
        key = ntm.memory[0:1].expand(batch_size, -1)
        strength = torch.ones(batch_size) * 10  # High strength for sharp attention
        
        weights = ntm._content_based_addressing(key, strength)
        
        # First slot should have highest weight
        assert weights[:, 0].mean() > weights[:, 1:].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
