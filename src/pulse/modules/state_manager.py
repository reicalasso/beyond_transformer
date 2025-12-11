"""State Manager for PULSEs.

This module implements dynamic state allocation and pruning mechanisms
for managing state nodes in PULSE architectures.
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateManager(nn.Module):
    """
    Manages state nodes with dynamic allocation and pruning capabilities.

    This class handles state node importance scoring and dynamic pruning
    based on learned importance scores.
    """

    def __init__(
        self,
        state_dim: int,
        max_states: int = 64,
        initial_states: Optional[int] = None,
        prune_threshold: float = 0.1,
    ) -> None:
        """
        Initialize the StateManager.

        Args:
            state_dim: Dimension of each state vector.
            max_states: Maximum number of state nodes.
            initial_states: Initial number of active states. Defaults to max_states.
            prune_threshold: Threshold for pruning states (0-1).
        """
        super(StateManager, self).__init__()
        self.state_dim = state_dim
        self.max_states = max_states
        self.prune_threshold = prune_threshold

        # Initialize state nodes
        initial_states = initial_states or max_states
        self.states = nn.Parameter(torch.randn(max_states, state_dim))

        # Learnable importance scores for each state node
        self.importance_scores = nn.Parameter(torch.ones(max_states))

        # Track active states
        self.register_buffer("active_mask", torch.ones(max_states, dtype=torch.bool))
        self.active_mask[initial_states:].fill_(False)

        # Initialize with small random values
        nn.init.normal_(self.states, mean=0.0, std=0.1)
        nn.init.uniform_(self.importance_scores, 0.5, 1.0)

    def forward(self, inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get current active states.

        Args:
            inputs: Optional input tensor for state updates (unused, for API compatibility).

        Returns:
            Active state vectors of shape [num_active_states, state_dim].
        """
        # Return only active states
        active_states = self.states[self.active_mask]
        return active_states

    def get_importance_scores(self) -> torch.Tensor:
        """
        Get importance scores for all states.

        Returns:
            Importance scores tensor of shape [max_states].
        """
        return torch.sigmoid(self.importance_scores)

    def get_active_count(self) -> int:
        """
        Get number of currently active states.

        Returns:
            Number of active states.
        """
        return self.active_mask.sum().item()

    def update_importance(
        self,
        gradients: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update importance scores based on gradients or attention weights.

        Args:
            gradients: Gradient information tensor.
            attention_weights: Attention weight information tensor.
        """
        # NOTE: These updates are not differentiable by design, as they represent
        # learned importance scores that are updated with exponential moving averages
        # rather than gradient descent. The parameter itself can still have gradients
        # from other operations in the computational graph.
        with torch.no_grad():
            if gradients is not None:
                # Update based on gradient magnitude
                grad_importance = torch.norm(gradients, dim=-1)
                # Exponential moving average update
                current_scores = torch.sigmoid(self.importance_scores)
                updated_scores = 0.9 * current_scores + 0.1 * grad_importance
                # Convert back to logits
                self.importance_scores.data = torch.log(
                    updated_scores / (1 - updated_scores + 1e-8)
                )

            if attention_weights is not None:
                # Update based on attention weights
                attention_importance = attention_weights.mean(
                    dim=0
                )  # Average across batch
                # Exponential moving average update
                current_scores = torch.sigmoid(self.importance_scores)
                updated_scores = 0.9 * current_scores + 0.1 * attention_importance
                # Convert back to logits
                self.importance_scores.data = torch.log(
                    updated_scores / (1 - updated_scores + 1e-8)
                )

    def prune_states(self, force_pruning: bool = False) -> int:
        """
        Prune states below importance threshold.

        Args:
            force_pruning: Force pruning even if it would reduce below minimum.

        Returns:
            Number of states pruned.
        """
        with torch.no_grad():
            importance_scores = self.get_importance_scores()
            low_importance_mask = importance_scores < self.prune_threshold

            # Don't prune if it would leave too few states
            current_active = self.active_mask.sum().item()
            would_prune = (self.active_mask & low_importance_mask).sum().item()

            if force_pruning or (current_active - would_prune) >= 1:
                # Update active mask
                self.active_mask = self.active_mask & ~low_importance_mask
                return would_prune
            else:
                return 0

    def prune_low_importance_states(self, force_pruning: bool = False) -> int:
        """
        Prune states below importance threshold.

        Alias for prune_states() for API compatibility.

        Args:
            force_pruning: Force pruning even if it would reduce below minimum.

        Returns:
            Number of states pruned.
        """
        return self.prune_states(force_pruning)

    def allocate_states(self, num_states: int) -> int:
        """
        Allocate additional state nodes.

        Args:
            num_states: Number of states to allocate.

        Returns:
            Number of states actually allocated.
        """
        with torch.no_grad():
            current_active = self.active_mask.sum().item()
            available_slots = self.max_states - current_active

            if available_slots <= 0:
                return 0

            num_to_allocate = min(num_states, available_slots)

            # Find inactive slots
            inactive_indices = torch.where(~self.active_mask)[0][:num_to_allocate]

            # Activate these slots
            self.active_mask[inactive_indices] = True

            # Initialize new states with small random values
            nn.init.normal_(self.states[inactive_indices], mean=0.0, std=0.1)

            return num_to_allocate

    def get_state_info(self) -> Dict[str, Union[int, float, List[float], List[int]]]:
        """
        Get detailed information about current states.

        Returns:
            Dictionary containing state information including total_states,
            active_states, importance_scores, active_indices, and prune_threshold.
        """
        importance_scores = self.get_importance_scores()
        active_indices = torch.where(self.active_mask)[0]

        return {
            "total_states": self.max_states,
            "active_states": self.get_active_count(),
            "importance_scores": importance_scores[self.active_mask].tolist(),
            "active_indices": active_indices.tolist(),
            "prune_threshold": self.prune_threshold,
        }


# Example usage
if __name__ == "__main__":
    # Create state manager
    state_manager = StateManager(state_dim=128, max_states=32, initial_states=16)

    # Get initial states
    states = state_manager()
    print(f"Initial active states: {states.shape}")

    # Check importance scores
    scores = state_manager.get_importance_scores()
    print(f"Importance scores range: {scores.min():.3f} - {scores.max():.3f}")

    # Prune states
    pruned = state_manager.prune_states()
    print(f"States pruned: {pruned}")
    print(f"Active states after pruning: {state_manager.get_active_count()}")

    # Allocate states
    allocated = state_manager.allocate_states(4)
    print(f"States allocated: {allocated}")
    print(f"Active states after allocation: {state_manager.get_active_count()}")

    # Get detailed info
    info = state_manager.get_state_info()
    print(f"State info: {info}")
