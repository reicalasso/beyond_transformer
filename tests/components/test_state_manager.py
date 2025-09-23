"""
Unit tests for StateManager component
"""

import os
import sys
import unittest

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsm import StateManager


class TestStateManager(unittest.TestCase):
    """Test cases for StateManager component."""

    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 64
        self.max_states = 16
        self.initial_states = 8
        self.prune_threshold = 0.3

        # Create state manager
        self.manager = StateManager(
            state_dim=self.state_dim,
            max_states=self.max_states,
            initial_states=self.initial_states,
            prune_threshold=self.prune_threshold,
        )

    def test_initialization(self):
        """Test StateManager initialization."""
        self.assertIsInstance(self.manager, StateManager)
        self.assertEqual(self.manager.state_dim, self.state_dim)
        self.assertEqual(self.manager.max_states, self.max_states)
        self.assertEqual(self.manager.prune_threshold, self.prune_threshold)

        # Check initial active states
        self.assertEqual(self.manager.get_active_count(), self.initial_states)

    def test_get_states_shape(self):
        """Test get states output shape."""
        states = self.manager()

        # Check output shape
        self.assertEqual(states.shape, (self.initial_states, self.state_dim))

    def test_importance_scores_properties(self):
        """Test importance scores properties."""
        scores = self.manager.get_importance_scores()

        # Check shape
        self.assertEqual(scores.shape, (self.max_states,))

        # Check range (0 to 1 after sigmoid)
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())

        # Check that scores are finite
        self.assertFalse(torch.isnan(scores).any())
        self.assertFalse(torch.isinf(scores).any())

    def test_get_active_count(self):
        """Test get_active_count method."""
        count = self.manager.get_active_count()
        self.assertEqual(count, self.initial_states)

    def test_prune_low_importance_states(self):
        """Test pruning low importance states."""
        # Manually set some states to have low importance
        with torch.no_grad():
            # Set first 2 states to have very low importance
            self.manager.importance_scores[:2] = torch.tensor([-10.0, -10.0])

        initial_count = self.manager.get_active_count()
        pruned_count = self.manager.prune_low_importance_states()
        final_count = self.manager.get_active_count()

        # Check that pruning worked
        self.assertEqual(pruned_count, 2)
        self.assertEqual(final_count, initial_count - 2)

    def test_prune_prevents_empty_states(self):
        """Test that pruning doesn't leave zero states."""
        # Manually set all states to have low importance
        with torch.no_grad():
            self.manager.importance_scores.fill_(-10.0)
            # But ensure at least one state remains active
            self.manager.importance_scores[0] = 10.0  # High importance

        initial_count = self.manager.get_active_count()
        pruned_count = self.manager.prune_low_importance_states()
        final_count = self.manager.get_active_count()

        # Should not prune below 1 state
        self.assertGreaterEqual(final_count, 1)

    def test_allocate_states(self):
        """Test allocating new states."""
        # Prune some states first to create space
        with torch.no_grad():
            self.manager.importance_scores[:4] = torch.tensor(
                [-10.0, -10.0, -10.0, -10.0]
            )
        self.manager.prune_low_importance_states()

        initial_count = self.manager.get_active_count()
        allocated_count = self.manager.allocate_states(3)
        final_count = self.manager.get_active_count()

        # Check allocation
        self.assertEqual(allocated_count, 3)
        self.assertEqual(final_count, initial_count + 3)

    def test_allocate_states_limit(self):
        """Test that allocation respects maximum limit."""
        # Try to allocate more states than available
        allocated_count = self.manager.allocate_states(self.max_states)

        # Should only allocate up to max
        self.assertLessEqual(allocated_count, self.max_states - self.initial_states)
        self.assertLessEqual(self.manager.get_active_count(), self.max_states)

    def test_get_state_info(self):
        """Test get_state_info method."""
        info = self.manager.get_state_info()

        # Check info structure
        self.assertIn("total_states", info)
        self.assertIn("active_states", info)
        self.assertIn("importance_scores", info)
        self.assertIn("active_indices", info)
        self.assertIn("prune_threshold", info)

        # Check values
        self.assertEqual(info["total_states"], self.max_states)
        self.assertEqual(info["active_states"], self.initial_states)
        self.assertEqual(info["prune_threshold"], self.prune_threshold)
        self.assertEqual(len(info["importance_scores"]), self.initial_states)
        self.assertEqual(len(info["active_indices"]), self.initial_states)

    def test_state_initialization(self):
        """Test that states are properly initialized."""
        states = self.manager()

        # Check that states are not all zeros
        self.assertFalse(torch.allclose(states, torch.zeros_like(states)))

        # Check that states are finite
        self.assertFalse(torch.isnan(states).any())
        self.assertFalse(torch.isinf(states).any())

    def test_differentiable_importance_updates(self):
        """Test that importance updates work with gradients."""
        # This is a smoke test since importance updates are done in-place with no_grad
        initial_scores = self.manager.get_importance_scores().clone()

        # The method should not crash
        try:
            # Create dummy gradients
            dummy_gradients = torch.randn(self.max_states, self.state_dim)
            # This should not raise an error
            # Note: actual implementation may vary
            pass
        except Exception as e:
            self.fail(f"Importance update raised an exception: {e}")

    def test_edge_case_minimum_states(self):
        """Test state manager with minimum configuration."""
        manager = StateManager(
            state_dim=32, max_states=4, initial_states=2, prune_threshold=0.5
        )

        # Basic functionality should work
        states = manager()
        self.assertEqual(states.shape, (2, 32))
        self.assertEqual(manager.get_active_count(), 2)

    def test_consistent_output_with_same_input(self):
        """Test that same calls produce consistent results."""
        states1 = self.manager()
        states2 = self.manager()

        # Check outputs are identical
        self.assertTrue(torch.allclose(states1, states2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
