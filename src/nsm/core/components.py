"""
Neural State Machine Components

This module implements core components for Neural State Machines including
TokenToStateRouter and enhanced StateManager.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenToStateRouter(nn.Module):
    """
    Token-to-State Router.

    Routes input tokens to appropriate state nodes based on learned routing mechanisms.
    """

    def __init__(
        self, token_dim: int, state_dim: int, num_states: int, num_heads: int = 4
    ):
        """
        Initialize the TokenToStateRouter.

        Args:
            token_dim (int): Dimension of input tokens
            state_dim (int): Dimension of state vectors
            num_states (int): Number of state nodes
            num_heads (int): Number of routing heads
        """
        super(TokenToStateRouter, self).__init__()
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.num_heads = num_heads

        # Routing mechanism
        self.router = nn.Linear(token_dim, num_states * num_heads)
        self.head_dim = state_dim // num_heads

        # Ensure compatibility
        assert state_dim % num_heads == 0, "state_dim must be divisible by num_heads"

        # Output projection
        self.output_projection = nn.Linear(state_dim, state_dim)

    def forward(
        self, tokens: torch.Tensor, states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to states.

        Args:
            tokens (torch.Tensor): Input tokens [batch_size, seq_len, token_dim]
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Routed tokens [batch_size, num_states, state_dim]
                - Routing weights [batch_size, seq_len, num_states]
        """
        batch_size, seq_len, token_dim = tokens.shape
        num_states = states.shape[1]

        # Compute routing weights
        routing_logits = self.router(
            tokens
        )  # [batch_size, seq_len, num_states * num_heads]
        routing_logits = routing_logits.view(
            batch_size, seq_len, self.num_heads, num_states
        )

        # Average across heads
        routing_weights = F.softmax(
            routing_logits.mean(dim=2), dim=-1
        )  # [batch_size, seq_len, num_states]

        # Route tokens to states
        # tokens: [batch_size, seq_len, token_dim]
        # routing_weights: [batch_size, seq_len, num_states]
        # Result: [batch_size, num_states, token_dim]
        routed_tokens = torch.bmm(routing_weights.transpose(1, 2), tokens)

        # Project to state dimension
        if token_dim != self.state_dim:
            routed_tokens = (
                F.pad(routed_tokens, (0, self.state_dim - token_dim))
                if token_dim < self.state_dim
                else routed_tokens[:, :, : self.state_dim]
            )

        # Apply output projection
        routed_tokens = self.output_projection(routed_tokens)

        return routed_tokens, routing_weights


class StateManager(nn.Module):
    """
    Enhanced State Manager with dynamic allocation capabilities.

    Manages state nodes with learnable importance scores and dynamic pruning.
    """

    def __init__(
        self,
        state_dim: int,
        max_states: int = 64,
        initial_states: Optional[int] = None,
        prune_threshold: float = 0.1,
    ):
        """
        Initialize the StateManager.

        Args:
            state_dim (int): Dimension of each state vector
            max_states (int): Maximum number of state nodes
            initial_states (int, optional): Initial number of active states
            prune_threshold (float): Threshold for pruning states (0-1)
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

        # Initialize parameters
        nn.init.normal_(self.states, mean=0.0, std=0.1)
        nn.init.uniform_(self.importance_scores, 0.5, 1.0)

    def forward(self) -> torch.Tensor:
        """
        Get current active states.

        Returns:
            torch.Tensor: Active state vectors [num_active_states, state_dim]
        """
        return self.states[self.active_mask]

    def get_importance_scores(self) -> torch.Tensor:
        """
        Get importance scores for all states.

        Returns:
            torch.Tensor: Importance scores [max_states]
        """
        return torch.sigmoid(self.importance_scores)

    def get_active_count(self) -> int:
        """
        Get number of currently active states.

        Returns:
            int: Number of active states
        """
        return self.active_mask.sum().item()

    def prune_low_importance_states(self) -> int:
        """
        Prune states below importance threshold.

        Returns:
            int: Number of states pruned
        """
        with torch.no_grad():
            importance_scores = self.get_importance_scores()
            low_importance_mask = (
                importance_scores < self.prune_threshold
            ) & self.active_mask

            # Don't prune if it would leave fewer than 1 state
            would_prune = low_importance_mask.sum().item()
            if self.get_active_count() - would_prune >= 1:
                # Avoid in-place operations that can cause gradient issues
                new_mask = self.active_mask.clone()
                new_mask[low_importance_mask] = False
                self.active_mask.copy_(new_mask)
                return would_prune
            else:
                return 0

    def allocate_states(self, num_states: int) -> int:
        """
        Allocate additional state nodes.

        Args:
            num_states (int): Number of states to allocate

        Returns:
            int: Number of states actually allocated
        """
        with torch.no_grad():
            current_active = self.get_active_count()
            available_slots = self.max_states - current_active

            if available_slots <= 0:
                return 0

            num_to_allocate = min(num_states, available_slots)

            # Find inactive slots
            inactive_indices = torch.where(~self.active_mask)[0][:num_to_allocate]

            # Activate these slots (avoid in-place operations)
            new_mask = self.active_mask.clone()
            new_mask[inactive_indices] = True
            self.active_mask.copy_(new_mask)

            # Initialize new states
            nn.init.normal_(self.states[inactive_indices], mean=0.0, std=0.1)

            return num_to_allocate

    def get_state_info(self) -> dict:
        """
        Get detailed information about current states.

        Returns:
            dict: State information
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
    # Test TokenToStateRouter
    batch_size, seq_len, token_dim = 2, 10, 64
    num_states, state_dim = 8, 128

    router = TokenToStateRouter(token_dim, state_dim, num_states)
    tokens = torch.randn(batch_size, seq_len, token_dim)
    states = torch.randn(batch_size, num_states, state_dim)

    routed_tokens, routing_weights = router(tokens, states)

    print(f"Input tokens shape: {tokens.shape}")
    print(f"States shape: {states.shape}")
    print(f"Routed tokens shape: {routed_tokens.shape}")
    print(f"Routing weights shape: {routing_weights.shape}")

    # Test StateManager
    state_manager = StateManager(state_dim=state_dim, max_states=16, initial_states=8)
    active_states = state_manager()

    print(f"\nInitial active states: {active_states.shape}")
    print(f"Active count: {state_manager.get_active_count()}")

    # Prune states
    pruned = state_manager.prune_low_importance_states()
    print(f"States pruned: {pruned}")
    print(f"Active count after pruning: {state_manager.get_active_count()}")

    # Allocate states
    allocated = state_manager.allocate_states(3)
    print(f"States allocated: {allocated}")
    print(f"Active count after allocation: {state_manager.get_active_count()}")

    print("Components test completed successfully!")
