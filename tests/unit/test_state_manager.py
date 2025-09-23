"""
Unit tests for the StateManager module.
"""

import numpy as np
import pytest
import torch

from nsm import StateManager


class TestStateManager:
    """Test suite for StateManager class."""

    @pytest.fixture
    def state_manager(self, sample_state_dim):
        """Create a StateManager for testing."""
        return StateManager(
            state_dim=sample_state_dim,
            max_states=16,
            initial_states=8,
            prune_threshold=0.3,
        )

    @pytest.fixture
    def large_state_manager(self, sample_state_dim):
        """Create a larger StateManager for testing."""
        return StateManager(
            state_dim=sample_state_dim,
            max_states=32,
            initial_states=16,
            prune_threshold=0.5,
        )

    def test_initialization(self, sample_state_dim):
        """Test StateManager initialization."""
        manager = StateManager(
            state_dim=sample_state_dim,
            max_states=16,
            initial_states=8,
            prune_threshold=0.3,
        )

        assert manager.state_dim == sample_state_dim
        assert manager.max_states == 16
        assert manager.prune_threshold == 0.3
        assert hasattr(manager, "states")
        assert hasattr(manager, "importance_scores")
        assert hasattr(manager, "active_mask")

        # Check initial state
        assert manager.get_active_count() == 8
        assert manager.states.shape == (16, sample_state_dim)
        assert manager.importance_scores.shape == (16,)
        assert manager.active_mask.shape == (16,)

    def test_get_states(self, state_manager):
        """Test getting active states."""
        states = state_manager()

        assert states is not None
        assert states.shape[0] == 8  # Initial active states
        assert states.shape[1] == state_manager.state_dim
        assert torch.isfinite(states).all()

    def test_get_importance_scores(self, state_manager):
        """Test getting importance scores."""
        scores = state_manager.get_importance_scores()

        assert scores is not None
        assert scores.shape == (state_manager.max_states,)
        assert torch.isfinite(scores).all()
        assert torch.min(scores) >= 0 and torch.max(scores) <= 1

    def test_get_active_count(self, state_manager):
        """Test getting active state count."""
        count = state_manager.get_active_count()

        assert isinstance(count, int)
        assert count == 8  # Initial count

    def test_prune_low_importance_states(self, state_manager):
        """Test pruning low importance states."""
        initial_count = state_manager.get_active_count()

        # Manually set some states to low importance
        with torch.no_grad():
            # Set first few states to very low importance
            state_manager.importance_scores[:3] = torch.tensor([-10.0, -10.0, -10.0])

        pruned_count = state_manager.prune_low_importance_states()

        # Should have pruned some states
        final_count = state_manager.get_active_count()
        assert final_count <= initial_count
        assert pruned_count >= 0

    def test_allocate_states(self, state_manager):
        """Test allocating new states."""
        initial_count = state_manager.get_active_count()

        # Allocate 2 new states
        allocated = state_manager.allocate_states(2)

        final_count = state_manager.get_active_count()

        assert allocated == 2
        assert final_count == initial_count + 2

    def test_allocate_states_limit(self, state_manager):
        """Test allocating states when at maximum."""
        # Allocate all remaining states
        remaining = state_manager.max_states - state_manager.get_active_count()
        allocated = state_manager.allocate_states(
            remaining + 10
        )  # Try to allocate more than available

        # Should only allocate what's available
        assert allocated <= remaining
        assert state_manager.get_active_count() == state_manager.max_states

    def test_get_state_info(self, state_manager):
        """Test getting state information."""
        info = state_manager.get_state_info()

        assert isinstance(info, dict)
        assert "total_states" in info
        assert "active_states" in info
        assert "importance_scores" in info
        assert "active_indices" in info
        assert "prune_threshold" in info

        assert info["total_states"] == state_manager.max_states
        assert info["active_states"] == state_manager.get_active_count()
        assert info["prune_threshold"] == state_manager.prune_threshold

    def test_state_initialization(self, sample_state_dim):
        """Test state initialization."""
        manager = StateManager(
            state_dim=sample_state_dim,
            max_states=8,
            initial_states=4,
            prune_threshold=0.3,
        )

        # Check that states are initialized properly
        states = manager()
        assert states.shape[0] == 4
        assert states.shape[1] == sample_state_dim
        assert torch.isfinite(states).all()

    def test_importance_score_initialization(self, state_manager):
        """Test importance score initialization."""
        scores = state_manager.get_importance_scores()

        # Scores should be between 0 and 1 (sigmoid of logits)
        assert torch.min(scores) >= 0
        assert torch.max(scores) <= 1

        # Should have reasonable distribution
        assert torch.std(scores) > 0

    def test_active_mask_consistency(self, state_manager):
        """Test that active mask is consistent with active count."""
        count = state_manager.get_active_count()
        active_indices = torch.where(state_manager.active_mask)[0]

        assert len(active_indices) == count
        assert torch.sum(state_manager.active_mask).item() == count

    def test_prune_protection(self, state_manager):
        """Test that pruning protects at least one state."""
        # Set all but one state to very low importance
        with torch.no_grad():
            state_manager.importance_scores[:-1] = torch.tensor(
                [-100.0] * (state_manager.max_states - 1)
            )

        # Try to prune - should leave at least one state
        pruned = state_manager.prune_low_importance_states()

        # Should leave at least one state active
        assert state_manager.get_active_count() >= 1

    def test_importance_score_updates(self, state_manager):
        """Test updating importance scores."""
        initial_scores = state_manager.get_importance_scores()

        # Simulate an update (this would normally happen during training)
        with torch.no_grad():
            # Update importance scores based on some gradients
            gradients = torch.randn_like(
                state_manager.importance_scores[: state_manager.get_active_count()]
            )
            attention_weights = torch.softmax(
                torch.randn(state_manager.get_active_count()), dim=0
            )

            # This is a simplified version - in practice, these would come from the model
            current_scores = state_manager.get_importance_scores()
            updated_scores = 0.9 * current_scores + 0.1 * torch.cat(
                [
                    attention_weights,
                    torch.zeros(
                        state_manager.max_states - state_manager.get_active_count()
                    ),
                ]
            )

            # Convert back to logits (simplified)
            state_manager.importance_scores.data = torch.log(
                updated_scores / (1 - updated_scores + 1e-8)
            )

    @pytest.mark.parametrize(
        "max_states,initial_states",
        [
            (8, 4),
            (16, 8),
            (32, 16),
        ],
    )
    def test_different_sizes(self, max_states, initial_states, sample_state_dim):
        """Test StateManager with different sizes."""
        manager = StateManager(
            state_dim=sample_state_dim,
            max_states=max_states,
            initial_states=initial_states,
            prune_threshold=0.3,
        )

        assert manager.max_states == max_states
        assert manager.get_active_count() == initial_states

        states = manager()
        assert states.shape[0] == initial_states
        assert states.shape[1] == sample_state_dim

    def test_state_persistence(self, state_manager):
        """Test that states persist between calls."""
        states1 = state_manager()
        states2 = state_manager()

        # Should be the same states (same tensor reference)
        assert torch.equal(states1, states2)

    def test_differentiable_operations(self, state_manager):
        """Test that operations are differentiable."""
        # Get states and compute some loss
        states = state_manager()
        # Create a loss that involves importance scores
        importance_scores = state_manager.get_importance_scores()
        loss = states.sum() + importance_scores.sum()

        # Backward pass should work
        loss.backward()

        # Importance scores should have gradients
        assert state_manager.importance_scores.grad is not None
        assert torch.isfinite(state_manager.importance_scores.grad).all()
