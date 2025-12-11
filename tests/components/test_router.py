"""
Unit tests for TokenToStateRouter component
"""

import os
import sys
import unittest

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse import TokenToStateRouter


class TestTokenToStateRouter(unittest.TestCase):
    """Test cases for TokenToStateRouter component."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 10
        self.token_dim = 32
        self.state_dim = 64
        self.num_states = 8
        self.num_heads = 4

        # Create router
        self.router = TokenToStateRouter(
            token_dim=self.token_dim,
            state_dim=self.state_dim,
            num_states=self.num_states,
            num_heads=self.num_heads,
        )

        # Create test data
        self.tokens = torch.randn(self.batch_size, self.seq_len, self.token_dim)
        self.states = torch.randn(self.batch_size, self.num_states, self.state_dim)

    def test_initialization(self):
        """Test TokenToStateRouter initialization."""
        self.assertIsInstance(self.router, TokenToStateRouter)
        self.assertEqual(self.router.token_dim, self.token_dim)
        self.assertEqual(self.router.state_dim, self.state_dim)
        self.assertEqual(self.router.num_states, self.num_states)
        self.assertEqual(self.router.num_heads, self.num_heads)

    def test_forward_pass_shapes(self):
        """Test forward pass output shapes."""
        routed_tokens, routing_weights = self.router(self.tokens, self.states)

        # Check routed tokens shape
        self.assertEqual(
            routed_tokens.shape, (self.batch_size, self.num_states, self.state_dim)
        )

        # Check routing weights shape
        self.assertEqual(
            routing_weights.shape, (self.batch_size, self.seq_len, self.num_states)
        )

    def test_routing_weights_sum_to_one(self):
        """Test that routing weights sum to 1 for each token."""
        routed_tokens, routing_weights = self.router(self.tokens, self.states)

        # Check that weights sum to 1 for each token
        weight_sums = routing_weights.sum(dim=-1)  # Sum over states
        expected_ones = torch.ones_like(weight_sums)

        self.assertTrue(torch.allclose(weight_sums, expected_ones, atol=1e-6))

    def test_forward_pass_differentiability(self):
        """Test that forward pass is differentiable."""
        self.tokens.requires_grad_(True)
        # Note: states are not used in the forward pass computation in TokenToStateRouter,
        # so they won't have gradients. Only tokens should have gradients.

        routed_tokens, routing_weights = self.router(self.tokens, self.states)
        loss = routed_tokens.sum() + routing_weights.sum()
        loss.backward()

        # Check gradients exist for tokens
        self.assertIsNotNone(self.tokens.grad)
        # States don't participate in the computation, so no gradient is expected

    def test_consistent_output_with_same_input(self):
        """Test that same input produces same output."""
        routed_tokens1, routing_weights1 = self.router(self.tokens, self.states)
        routed_tokens2, routing_weights2 = self.router(self.tokens, self.states)

        # Check outputs are identical
        self.assertTrue(torch.allclose(routed_tokens1, routed_tokens2, atol=1e-6))
        self.assertTrue(torch.allclose(routing_weights1, routing_weights2, atol=1e-6))

    def test_routing_weights_are_probabilities(self):
        """Test that routing weights are valid probabilities."""
        routed_tokens, routing_weights = self.router(self.tokens, self.states)

        # Check that weights are non-negative
        self.assertTrue((routing_weights >= 0).all())

        # Check that weights are finite
        self.assertFalse(torch.isnan(routing_weights).any())
        self.assertFalse(torch.isinf(routing_weights).any())

    def test_different_token_and_state_dimensions(self):
        """Test router with different token and state dimensions."""
        # Create router with different dimensions
        router = TokenToStateRouter(
            token_dim=16,  # Different from state_dim
            state_dim=64,
            num_states=4,
            num_heads=2,
        )

        tokens = torch.randn(self.batch_size, self.seq_len, 16)
        states = torch.randn(self.batch_size, 4, 64)

        routed_tokens, routing_weights = router(tokens, states)

        # Check shapes
        self.assertEqual(routed_tokens.shape, (self.batch_size, 4, 64))
        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, 4))

    def test_edge_case_single_state(self):
        """Test router with single state."""
        router = TokenToStateRouter(
            token_dim=self.token_dim,
            state_dim=self.state_dim,
            num_states=1,  # Single state
            num_heads=self.num_heads,
        )

        tokens = torch.randn(self.batch_size, self.seq_len, self.token_dim)
        states = torch.randn(self.batch_size, 1, self.state_dim)

        routed_tokens, routing_weights = router(tokens, states)

        # Check shapes
        self.assertEqual(routed_tokens.shape, (self.batch_size, 1, self.state_dim))
        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, 1))

        # Check weights sum to 1
        weight_sums = routing_weights.sum(dim=-1)
        expected_ones = torch.ones_like(weight_sums)
        self.assertTrue(torch.allclose(weight_sums, expected_ones, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
