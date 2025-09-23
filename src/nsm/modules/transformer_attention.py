"""
Transformer Attention Layer Implementation

This module implements a standard Transformer attention layer that can be used
as a component in the Neural State Machine architecture.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerAttention(nn.Module):
    """
    Standard Transformer Attention Layer.

    This module implements multi-head self-attention mechanism as used in
    the original Transformer architecture.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        """
        Initialize the TransformerAttention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(TransformerAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = self.d_k**-0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerAttention.

        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_k, d_model]
            attention_mask: Attention mask [seq_len_q, seq_len_k] or [batch_size, seq_len_q, seq_len_k]
            key_padding_mask: Key padding mask [batch_size, seq_len_k]

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len_q, d_model]
            - attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query)  # [batch_size, seq_len_q, d_model]
        k = self.k_proj(key)  # [batch_size, seq_len_k, d_model]
        v = self.v_proj(value)  # [batch_size, seq_len_k, d_model]

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_q, d_k]
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_k, d_k]
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(
            1, 2
        )  # [batch_size, num_heads, seq_len_k, d_k]

        # Compute attention scores
        scores = (
            torch.matmul(q, k.transpose(-2, -1)) * self.scale
        )  # [batch_size, num_heads, seq_len_q, seq_len_k]

        # Apply masks
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(
                    0
                )  # [1, 1, seq_len_q, seq_len_k]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(
                    1
                )  # [batch_size, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(
                2
            )  # [batch_size, 1, 1, seq_len_k]
            scores = scores.masked_fill(key_padding_mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(
            scores, dim=-1
        )  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attention_weights = self.attn_dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(
            attention_weights, v
        )  # [batch_size, num_heads, seq_len_q, d_k]

        # Reshape and project output
        context = context.transpose(
            1, 2
        ).contiguous()  # [batch_size, seq_len_q, num_heads, d_k]
        context = context.view(
            batch_size, -1, self.d_model
        )  # [batch_size, seq_len_q, d_model]
        output = self.out_proj(context)  # [batch_size, seq_len_q, d_model]
        output = self.proj_dropout(output)

        return output, attention_weights

    def forward_self_attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [seq_len, seq_len] or [batch_size, seq_len, seq_len]
            key_padding_mask: Key padding mask [batch_size, seq_len]

        Returns:
            Tuple of (output, attention_weights)
        """
        return self.forward(x, x, x, attention_mask, key_padding_mask)


# Example usage
if __name__ == "__main__":
    # Test TransformerAttention
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8

    # Create TransformerAttention
    transformer_attn = TransformerAttention(d_model=d_model, num_heads=num_heads)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass (self-attention)
    output, attn_weights = transformer_attn.forward_self_attention(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print("TransformerAttention test completed successfully!")
