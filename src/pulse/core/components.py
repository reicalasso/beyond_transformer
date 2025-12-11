"""
PULSE Components

This module implements core components for PULSEs including
TokenToStateRouter. StateManager is imported from modules.state_manager for consistency.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import StateManager from the canonical location to avoid duplication
from ..modules.state_manager import StateManager

__all__ = ["TokenToStateRouter", "StateManager"]


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
