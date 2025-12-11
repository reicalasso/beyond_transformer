"""
Optimized Attention Mechanisms for PULSE.

This module provides highly optimized attention implementations:
- FlashAttention-style memory-efficient attention
- Fused operations for state attention
- Chunked processing for long sequences
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedStateAttention(nn.Module):
    """
    Optimized state attention with fused operations.
    
    Key optimizations:
    - Fused QKV projection
    - In-place operations where possible
    - Efficient memory layout
    - Optional chunked processing for long sequences
    - Flash attention style memory efficiency
    """

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        chunk_size: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5

        assert hidden_size % num_heads == 0

        # Fused QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv_proj = nn.Linear(state_dim, hidden_size * 2, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Check for flash attention availability
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            states: [batch, num_states, state_dim]
            attention_mask: Optional mask

        Returns:
            output: [batch, seq_len, hidden_size]
            attention_weights: [batch, num_heads, seq_len, num_states]
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_states = states.shape[1]

        # Project queries from hidden states
        q = self.q_proj(hidden_states)
        
        # Fused KV projection from states
        kv = self.kv_proj(states)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_states, self.num_heads, self.head_dim).transpose(1, 2)

        # Use PyTorch 2.0 flash attention if available
        if self.use_flash and attention_mask is None:
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        elif seq_len > self.chunk_size:
            output = self._chunked_attention(q, k, v, attention_mask)
        else:
            output = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)

        # Return dummy attention weights (computing them adds overhead)
        return output, None

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _chunked_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Memory-efficient chunked attention for long sequences."""
        batch_size, num_heads, seq_len, head_dim = q.shape
        num_states = k.shape[2]

        outputs = []
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_idx, :]

            # Compute attention for this chunk
            attn_weights = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                chunk_mask = mask[:, :, i:end_idx, :] if mask.dim() == 4 else mask
                attn_weights = attn_weights.masked_fill(chunk_mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)


class OptimizedSelfAttention(nn.Module):
    """
    Optimized self-attention with causal masking.
    
    Optimizations:
    - Fused QKV projection
    - Efficient causal mask generation
    - Optional KV caching for inference
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        assert hidden_size % num_heads == 0

        # Fused QKV projection (3x hidden_size)
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Pre-compute causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
            persistent=False,
        )
        
        # Check for flash attention
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            kv_cache: Optional tuple of (cached_k, cached_v)

        Returns:
            output: [batch, seq_len, hidden_size]
            new_kv_cache: Updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)
        kv_seq_len = k.shape[2]

        # Use flash attention if available and no KV cache
        if self.use_flash and kv_cache is None:
            output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            # Compute attention
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply causal mask
            if seq_len > 1:
                causal_mask = self.causal_mask[:seq_len, :kv_seq_len]
                attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)

            output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)

        return output, new_kv_cache


class OptimizedStatePropagator(nn.Module):
    """
    Optimized state propagator with fused GRU operations.
    
    Key optimizations:
    - Fused gate computations
    - Vectorized batch processing
    - Reduced memory allocations
    """

    def __init__(
        self,
        state_dim: int,
        gate_type: str = "gru",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.gate_type = gate_type

        if gate_type == "gru":
            # Fused gate projection: [reset, update, candidate] = 3 * state_dim
            self.gate_proj = nn.Linear(state_dim * 2, state_dim * 3, bias=True)
        else:  # lstm
            # Fused gate projection: [forget, input, output, candidate] = 4 * state_dim
            self.gate_proj = nn.Linear(state_dim * 2, state_dim * 4, bias=True)

    def forward(
        self,
        states: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized state update.

        Args:
            states: [batch, num_states, state_dim]
            inputs: [batch, num_states, state_dim]

        Returns:
            Updated states: [batch, num_states, state_dim]
        """
        batch_size, num_states, state_dim = states.shape

        # Concatenate states and inputs
        combined = torch.cat([states, inputs], dim=-1)

        if self.gate_type == "gru":
            return self._gru_update(states, combined)
        else:
            return self._lstm_update(states, combined)

    def _gru_update(
        self,
        states: torch.Tensor,
        combined: torch.Tensor,
    ) -> torch.Tensor:
        """Fused GRU update."""
        # Single projection for all gates
        gates = self.gate_proj(combined)
        reset, update, candidate = gates.chunk(3, dim=-1)

        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)

        # Candidate with reset gate applied
        reset_states = reset * states
        candidate = torch.tanh(candidate + reset_states - states)  # Approximate reset effect

        # Update states
        new_states = (1 - update) * states + update * candidate

        return new_states

    def _lstm_update(
        self,
        states: torch.Tensor,
        combined: torch.Tensor,
    ) -> torch.Tensor:
        """Fused LSTM update."""
        gates = self.gate_proj(combined)
        forget, input_gate, output, candidate = gates.chunk(4, dim=-1)

        forget = torch.sigmoid(forget)
        input_gate = torch.sigmoid(input_gate)
        output = torch.sigmoid(output)
        candidate = torch.tanh(candidate)

        # Update cell state
        cell = forget * states + input_gate * candidate

        # Output
        new_states = output * torch.tanh(cell)

        return new_states


class OptimizedPulseLayer(nn.Module):
    """
    Fully optimized PULSE layer combining all optimizations.
    """

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        num_heads: int = 8,
        num_states: int = 16,
        intermediate_size: int = None,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        intermediate_size = intermediate_size or hidden_size * 4

        # Optimized attention modules
        self.self_attention = OptimizedSelfAttention(hidden_size, num_heads, dropout)
        self.state_attention = OptimizedStateAttention(hidden_size, state_dim, num_heads, dropout)
        self.state_propagator = OptimizedStatePropagator(state_dim)

        # Layer norms
        self.attention_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # FFN with fused gating (SwiGLU-style)
        self.ffn_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.ffn_down = nn.Linear(intermediate_size, hidden_size, bias=False)

        # State projection
        self.state_proj = nn.Linear(hidden_size, state_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Optimized forward pass.
        """
        # Self-attention
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, new_kv_cache = self.self_attention(hidden_states, kv_cache)
        hidden_states = residual + hidden_states

        # State attention
        residual = hidden_states
        hidden_states = self.state_norm(hidden_states)
        state_output, _ = self.state_attention(hidden_states, states)
        hidden_states = residual + state_output

        # Update states - use efficient pooling
        state_input = hidden_states.mean(dim=1, keepdim=True)
        state_input = self.state_proj(state_input).expand(-1, states.shape[1], -1)
        updated_states = self.state_propagator(states, state_input)

        # FFN with SwiGLU
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        gate = F.silu(self.ffn_gate(hidden_states))
        up = self.ffn_up(hidden_states)
        hidden_states = residual + self.ffn_down(gate * up)

        return hidden_states, updated_states, new_kv_cache
