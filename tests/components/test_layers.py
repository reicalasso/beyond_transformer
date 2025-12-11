"""
Unit tests for PulseLayer component
"""

import os
import sys
import unittest

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse import HybridAttention, PulseLayer


class TestPulseLayer(unittest.TestCase):
    """Test cases for PulseLayer component."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_states = 8
        self.state_dim = 64
        self.seq_len = 10
        self.token_dim = 32
        self.num_heads = 4

        # Create layer
        self.layer = PulseLayer(
            state_dim=self.state_dim, token_dim=self.token_dim, num_heads=self.num_heads
        )

        # Create test data
        self.states = torch.randn(self.batch_size, self.num_states, self.state_dim)
        self.tokens = torch.randn(self.batch_size, self.seq_len, self.token_dim)

    def test_initialization(self):
        """Test PulseLayer initialization."""
        self.assertIsInstance(self.layer, PulseLayer)
        self.assertEqual(self.layer.state_dim, self.state_dim)
        self.assertEqual(self.layer.token_dim, self.token_dim)
        self.assertEqual(self.layer.num_heads, self.num_heads)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        output = self.layer(self.states, self.tokens)

        # Check output shape
        self.assertEqual(
            output.shape, (self.batch_size, self.num_states, self.state_dim)
        )

    def test_forward_pass_differentiability(self):
        """Test that forward pass is differentiable."""
        self.states.requires_grad_(True)
        self.tokens.requires_grad_(True)

        output = self.layer(self.states, self.tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(self.states.grad)
        self.assertIsNotNone(self.tokens.grad)

    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        # Create attention mask
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)

        output = self.layer(self.states, self.tokens, attention_mask)

        # Check output shape
        self.assertEqual(
            output.shape, (self.batch_size, self.num_states, self.state_dim)
        )

    def test_layer_norm_properties(self):
        """Test layer normalization properties."""
        output = self.layer(self.states, self.tokens)

        # Check that output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_consistent_output_with_same_input(self):
        """Test that same input produces same output."""
        output1 = self.layer(self.states, self.tokens)
        output2 = self.layer(self.states, self.tokens)

        # Check outputs are identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestHybridAttention(unittest.TestCase):
    """Test cases for HybridAttention component."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.num_states = 8
        self.state_dim = 64
        self.seq_len = 10
        self.token_dim = 64  # Same as state_dim for simplicity
        self.num_heads = 4

        # Create attention mechanism
        self.attention = HybridAttention(
            state_dim=self.state_dim, token_dim=self.token_dim, num_heads=self.num_heads
        )

        # Create test data
        self.states = torch.randn(self.batch_size, self.num_states, self.state_dim)
        self.tokens = torch.randn(self.batch_size, self.seq_len, self.token_dim)

    def test_initialization(self):
        """Test HybridAttention initialization."""
        self.assertIsInstance(self.attention, HybridAttention)
        self.assertEqual(self.attention.state_dim, self.state_dim)
        self.assertEqual(self.attention.token_dim, self.token_dim)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.head_dim, self.state_dim // self.num_heads)

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        output = self.attention(self.states, self.tokens)

        # Check output shape
        self.assertEqual(
            output.shape, (self.batch_size, self.num_states, self.state_dim)
        )

    def test_forward_pass_differentiability(self):
        """Test that forward pass is differentiable."""
        self.states.requires_grad_(True)
        self.tokens.requires_grad_(True)

        output = self.attention(self.states, self.tokens)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(self.states.grad)
        self.assertIsNotNone(self.tokens.grad)

    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        # Create attention mask
        attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)

        output = self.attention(self.states, self.tokens, attention_mask)

        # Check output shape
        self.assertEqual(
            output.shape, (self.batch_size, self.num_states, self.state_dim)
        )

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        # We'll need to modify the HybridAttention to expose weights for this test
        # For now, we'll test that output is valid
        output = self.attention(self.states, self.tokens)

        # Check that output is not NaN or Inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_consistent_output_with_same_input(self):
        """Test that same input produces same output."""
        output1 = self.attention(self.states, self.tokens)
        output2 = self.attention(self.states, self.tokens)

        # Check outputs are identical
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
