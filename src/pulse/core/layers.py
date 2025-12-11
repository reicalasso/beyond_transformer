"""
PULSE Layers

This module implements core layers for PULSEs including
PulseLayer and HybridAttention mechanisms.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseLayer(nn.Module):
    """
    PULSE Layer.

    This layer implements a single step of state machine computation,
    combining state updates with token-to-state routing.
    """

    def __init__(self, state_dim: int, token_dim: int, num_heads: int = 4):
        """
        Initialize the PulseLayer.

        Args:
            state_dim (int): Dimension of state vectors
            token_dim (int): Dimension of input tokens
            num_heads (int): Number of attention heads
        """
        super(PulseLayer, self).__init__()
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.num_heads = num_heads

        # State update mechanisms
        self.state_update = nn.Linear(
            state_dim + state_dim, state_dim
        )  # states + attended_tokens

        # Hybrid attention mechanism
        self.hybrid_attention = HybridAttention(state_dim, token_dim, num_heads)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the PulseLayer.

        Args:
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]
            tokens (torch.Tensor): Input tokens [batch_size, seq_len, token_dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Updated states [batch_size, num_states, state_dim]
        """
        batch_size, num_states, state_dim = states.shape
        seq_len = tokens.shape[1]

        # Apply hybrid attention to route tokens to states
        attended_tokens = self.hybrid_attention(states, tokens, attention_mask)

        # Update states based on attended tokens
        # Use attended tokens as input for state update
        state_updates = self.state_update(torch.cat([states, attended_tokens], dim=-1))

        # Apply gated update
        updated_states = self.layer_norm(states + state_updates)

        return updated_states


class HybridAttention(nn.Module):
    """
    Hybrid Attention Mechanism.

    Combines token-to-state routing with content-based attention.
    """

    def __init__(self, state_dim: int, token_dim: int, num_heads: int = 4):
        """
        Initialize the HybridAttention.

        Args:
            state_dim (int): Dimension of state vectors
            token_dim (int): Dimension of input tokens
            num_heads (int): Number of attention heads
        """
        super(HybridAttention, self).__init__()
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = state_dim // num_heads

        # Ensure head dimension is compatible
        assert state_dim % num_heads == 0, "state_dim must be divisible by num_heads"

        # Linear projections
        self.state_projection = nn.Linear(state_dim, state_dim)
        self.token_projection = nn.Linear(token_dim, state_dim)
        self.output_projection = nn.Linear(state_dim, state_dim)

        # Attention scaling factor
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of HybridAttention.

        Args:
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]
            tokens (torch.Tensor): Input tokens [batch_size, seq_len, token_dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]

        Returns:
            torch.Tensor: Attended tokens [batch_size, num_states, state_dim]
        """
        batch_size, num_states, state_dim = states.shape
        seq_len = tokens.shape[1]

        # Linear projections
        states_proj = self.state_projection(
            states
        )  # [batch_size, num_states, state_dim]
        tokens_proj = self.token_projection(tokens)  # [batch_size, seq_len, state_dim]

        # Reshape for multi-head attention
        states_proj = states_proj.view(
            batch_size, num_states, self.num_heads, self.head_dim
        )
        tokens_proj = tokens_proj.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        states_proj = states_proj.transpose(
            1, 2
        )  # [batch_size, num_heads, num_states, head_dim]
        tokens_proj = tokens_proj.transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        # Q @ K^T where Q=states, K=tokens
        attention_scores = torch.matmul(states_proj, tokens_proj.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(
                ~attention_mask, float("-inf")
            )

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values (tokens)
        attended_tokens = torch.matmul(attention_weights, tokens_proj)

        # Transpose back and reshape
        attended_tokens = attended_tokens.transpose(1, 2).contiguous()
        attended_tokens = attended_tokens.view(batch_size, num_states, state_dim)

        # Final linear projection
        output = self.output_projection(attended_tokens)

        return output


# Example usage
if __name__ == "__main__":
    # Test PulseLayer
    batch_size, num_states, state_dim = 2, 8, 128
    seq_len, token_dim = 10, 64

    # Create layer
    pulse_layer = PulseLayer(state_dim=state_dim, token_dim=token_dim, num_heads=4)

    # Create sample inputs
    states = torch.randn(batch_size, num_states, state_dim)
    tokens = torch.randn(batch_size, seq_len, token_dim)

    # Forward pass
    updated_states = pulse_layer(states, tokens)

    print(f"Input states shape: {states.shape}")
    print(f"Input tokens shape: {tokens.shape}")
    print(f"Output states shape: {updated_states.shape}")

    # Test HybridAttention
    hybrid_attention = HybridAttention(
        state_dim=state_dim, token_dim=token_dim, num_heads=4
    )
    attended_tokens = hybrid_attention(states, tokens)

    print(f"Attended tokens shape: {attended_tokens.shape}")
    print("Layers test completed successfully!")
