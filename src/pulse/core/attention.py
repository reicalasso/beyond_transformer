"""
Advanced Attention Mechanisms for PULSEs.

This module implements various attention mechanisms optimized for PULSE architecture:
- SparseStateAttention: Efficient sparse attention between tokens and states
- MultiScaleAttention: Hierarchical attention at multiple temporal scales
- LinearAttention: O(n) complexity attention using kernel approximation
- CausalStateAttention: Causal attention for autoregressive generation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseStateAttention(nn.Module):
    """
    Sparse attention mechanism for efficient token-to-state routing.
    
    Uses top-k selection to route each token to only the most relevant states,
    achieving O(n * k) complexity where k << num_states.
    """

    def __init__(
        self,
        token_dim: int,
        state_dim: int,
        num_heads: int = 8,
        top_k: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize SparseStateAttention.

        Args:
            token_dim: Dimension of input tokens.
            state_dim: Dimension of state vectors.
            num_heads: Number of attention heads.
            top_k: Number of states each token attends to.
            dropout: Dropout probability.
        """
        super().__init__()
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.top_k = top_k
        self.head_dim = state_dim // num_heads

        assert state_dim % num_heads == 0, "state_dim must be divisible by num_heads"

        # Projections
        self.q_proj = nn.Linear(token_dim, state_dim)
        self.k_proj = nn.Linear(state_dim, state_dim)
        self.v_proj = nn.Linear(state_dim, state_dim)
        self.out_proj = nn.Linear(state_dim, state_dim)

        # Routing network for top-k selection
        self.router = nn.Linear(token_dim, state_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        tokens: torch.Tensor,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse attention.

        Args:
            tokens: Input tokens [batch_size, seq_len, token_dim]
            states: State vectors [batch_size, num_states, state_dim]
            attention_mask: Optional mask [batch_size, seq_len]

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch_size, seq_len, state_dim]
            - attention_weights: [batch_size, num_heads, seq_len, top_k]
        """
        batch_size, seq_len, _ = tokens.shape
        num_states = states.shape[1]

        # Compute routing scores for top-k selection
        routing_scores = torch.matmul(
            self.router(tokens), states.transpose(-2, -1)
        )  # [batch, seq_len, num_states]

        # Select top-k states for each token
        top_k_scores, top_k_indices = torch.topk(
            routing_scores, min(self.top_k, num_states), dim=-1
        )  # [batch, seq_len, top_k]

        # Gather selected states
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(
            -1, -1, -1, self.state_dim
        )
        selected_states = torch.gather(
            states.unsqueeze(1).expand(-1, seq_len, -1, -1),
            dim=2,
            index=top_k_indices_expanded,
        )  # [batch, seq_len, top_k, state_dim]

        # Project queries, keys, values
        q = self.q_proj(tokens)  # [batch, seq_len, state_dim]
        k = self.k_proj(selected_states)  # [batch, seq_len, top_k, state_dim]
        v = self.v_proj(selected_states)  # [batch, seq_len, top_k, state_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.top_k, self.num_heads, self.head_dim)
        k = k.permute(0, 3, 1, 2, 4)  # [batch, heads, seq_len, top_k, head_dim]
        v = v.view(batch_size, seq_len, self.top_k, self.num_heads, self.head_dim)
        v = v.permute(0, 3, 1, 2, 4)

        # Compute attention scores
        q = q.unsqueeze(-2)  # [batch, heads, seq_len, 1, head_dim]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(-2)
        attn_scores = attn_scores * self.scale  # [batch, heads, seq_len, top_k]

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        output = (attn_weights_expanded * v).sum(dim=-2)  # [batch, heads, seq_len, head_dim]

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.state_dim)
        output = self.out_proj(output)

        return output, attn_weights


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) complexity.
    
    Uses kernel feature maps to approximate softmax attention,
    enabling linear scaling with sequence length.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        feature_dim: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize LinearAttention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            feature_dim: Dimension of kernel features.
            dropout: Dropout probability.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.feature_dim = feature_dim
        self.eps = eps

        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Learnable feature map parameters
        self.feature_map = nn.Parameter(torch.randn(self.head_dim, feature_dim))

        self.dropout = nn.Dropout(dropout)
        self.scale = feature_dim ** -0.5

    def _feature_map_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map for kernel approximation."""
        # ELU + 1 feature map (ensures positivity)
        return F.elu(x) + 1

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with linear attention.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply feature map
        q = self._feature_map_fn(q)
        k = self._feature_map_fn(k)

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).unsqueeze(-1)
            k = k * mask
            v = v * mask

        # Linear attention: O(n) complexity
        # Instead of Q @ K^T @ V (O(n^2)), compute K^T @ V first (O(n))
        kv = torch.einsum('bshd,bshm->bhdm', k, v)  # [batch, heads, head_dim, head_dim]
        qkv = torch.einsum('bshd,bhdm->bshm', q, kv)  # [batch, seq, heads, head_dim]

        # Normalize
        k_sum = k.sum(dim=1, keepdim=True)  # [batch, 1, heads, head_dim]
        normalizer = torch.einsum('bshd,bthd->bsh', q, k_sum) + self.eps
        output = qkv / normalizer.unsqueeze(-1)

        # Reshape and project
        output = output.reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output


class CausalStateAttention(nn.Module):
    """
    Causal attention for autoregressive state updates.
    
    Ensures that each position can only attend to previous positions,
    enabling autoregressive generation with PULSE.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        """
        Initialize CausalStateAttention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            max_seq_len: Maximum sequence length for causal mask.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len

        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Register causal mask as buffer
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with causal attention.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            kv_cache: Optional cached key-value pairs for incremental decoding

        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_kv_cache = (k, v)
        kv_len = k.shape[1]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if kv_cache is None:
            # Full sequence - use pre-computed causal mask
            mask = self.causal_mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Incremental decoding - only mask future positions
            start_pos = kv_len - seq_len
            mask = self.causal_mask[start_pos:start_pos + seq_len, :kv_len]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)

        return output, new_kv_cache


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention for capturing patterns at different temporal scales.
    
    Combines local attention (fine-grained) with global attention (coarse-grained)
    for efficient long-range dependency modeling.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        local_window: int = 64,
        global_tokens: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize MultiScaleAttention.

        Args:
            dim: Model dimension.
            num_heads: Number of attention heads.
            local_window: Size of local attention window.
            global_tokens: Number of global summary tokens.
            dropout: Dropout probability.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_window = local_window
        self.global_tokens = global_tokens

        assert dim % num_heads == 0

        # Local attention
        self.local_q = nn.Linear(dim, dim)
        self.local_k = nn.Linear(dim, dim)
        self.local_v = nn.Linear(dim, dim)

        # Global attention
        self.global_tokens_param = nn.Parameter(torch.randn(1, global_tokens, dim))
        self.global_q = nn.Linear(dim, dim)
        self.global_k = nn.Linear(dim, dim)
        self.global_v = nn.Linear(dim, dim)

        # Output projection
        self.out_proj = nn.Linear(dim * 2, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Layer norms
        self.local_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-scale attention.

        Args:
            x: Input tensor [batch_size, seq_len, dim]

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape

        # Local attention (windowed)
        local_output = self._local_attention(x)

        # Global attention (with summary tokens)
        global_output = self._global_attention(x)

        # Combine local and global
        combined = torch.cat([local_output, global_output], dim=-1)
        output = self.out_proj(combined)

        return output

    def _local_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute local windowed attention."""
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.local_q(x)
        k = self.local_k(x)
        v = self.local_v(x)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention with local window mask
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create local window mask
        positions = torch.arange(seq_len, device=x.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        local_mask = distance > (self.local_window // 2)
        attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.local_norm(output)

    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute global attention with summary tokens."""
        batch_size, seq_len, _ = x.shape

        # Expand global tokens for batch
        global_tokens = self.global_tokens_param.expand(batch_size, -1, -1)

        # Global tokens attend to all positions
        q = self.global_q(global_tokens)
        k = self.global_k(x)
        v = self.global_v(x)

        # Reshape for multi-head
        q = q.view(batch_size, self.global_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        global_summary = torch.matmul(attn_weights, v)
        global_summary = global_summary.transpose(1, 2).contiguous().view(
            batch_size, self.global_tokens, self.dim
        )

        # Broadcast global summary back to sequence positions
        # Each position gets a weighted combination of global summaries
        q_seq = self.global_q(x)
        k_global = self.global_k(global_summary)
        v_global = self.global_v(global_summary)

        q_seq = q_seq.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_global = k_global.view(batch_size, self.global_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v_global = v_global.view(batch_size, self.global_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q_seq, k_global.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, v_global)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        return self.global_norm(output)
