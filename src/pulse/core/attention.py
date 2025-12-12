"""
PULSE Attention Mechanisms

Provides various attention implementations:
- GroupedQueryAttention (GQA): Memory-efficient with fewer KV heads
- MultiHeadAttention: Standard implementation
- All support RoPE, KV caching, and Flash Attention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).
    
    Uses fewer KV heads than query heads for memory efficiency.
    - num_heads query heads
    - num_kv_heads key/value heads (num_heads must be divisible by num_kv_heads)
    
    Used in Llama 2 70B, Mistral, and other large models.
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (default: num_heads // 4)
        head_dim: Dimension per head (default: hidden_size // num_heads)
        dropout: Attention dropout
        max_position_embeddings: Max sequence length for RoPE
        use_rope: Whether to use rotary embeddings
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int = None,
        head_dim: int = None,
        dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        use_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads // 4 or 1
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.use_rope = use_rope
        
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})"
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE
        if use_rope:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)
        else:
            self.rotary_emb = None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Flash attention check
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_value: Optional KV cache (k, v)
            use_cache: Whether to return updated cache
            
        Returns:
            output: [batch, seq_len, hidden_size]
            new_cache: Optional updated KV cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        new_cache = (k, v) if use_cache else None
        kv_seq_len = k.shape[2]
        
        # Expand KV for GQA (repeat KV heads to match Q heads)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        if self.use_flash and attention_mask is None and past_key_value is None:
            # Use Flash Attention
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0 if not self.training else 0.0
            )
        else:
            # Standard attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask
            if attention_mask is None and seq_len > 1:
                causal_mask = torch.triu(
                    torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=q.device),
                    diagonal=kv_seq_len - seq_len + 1
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            elif attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_cache


class MultiHeadAttention(GroupedQueryAttention):
    """
    Standard Multi-Head Attention.
    
    Equivalent to GQA with num_kv_heads = num_heads.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        use_rope: bool = True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,  # Same as num_heads for MHA
            head_dim=head_dim,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
            use_rope=use_rope,
        )


class StateAttention(nn.Module):
    """
    Attention from hidden states to memory states.
    
    Queries come from hidden states, keys/values from states.
    No causal masking needed.
    
    Args:
        hidden_size: Hidden dimension
        state_dim: State dimension
        num_heads: Number of attention heads
        dropout: Attention dropout
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q from hidden, KV from states
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(state_dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            states: [batch, num_states, state_dim]
            
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_states = states.shape[1]
        
        # Project
        q = self.q_proj(hidden_states)
        k = self.k_proj(states)
        v = self.v_proj(states)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_states, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention (no causal mask for state attention)
        if self.use_flash:
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output

