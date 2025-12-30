"""
PULSE Unified Block

Single primitive combining:
- Local patterns: Depthwise convolution
- Global context: Linear attention with decay
- Gated fusion

O(n) complexity, replaces separate SSM/Attention/State modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class LinearAttention(nn.Module):
    """
    Linear Attention with exponential decay.
    
    O(n) complexity via kernel trick:
    - Q, K projected through feature map (elu + 1)
    - Cumulative sum instead of full attention matrix
    - Exponential decay for recency bias
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        decay: Exponential decay factor (0.9-0.99)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        decay: float = 0.95,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.decay = decay
        
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Learnable decay per head
        self.decay_param = nn.Parameter(torch.ones(num_heads) * decay)
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional recurrent state.
        
        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional [batch, num_heads, head_dim, head_dim] recurrent state
            
        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: [batch, num_heads, head_dim, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Feature map: elu(x) + 1 for non-negative keys/queries
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention with decay
        output, new_state = self._linear_attention(q, k, v, state)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)
        
        return output, new_state
    
    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute linear attention with cumulative state.
        
        Uses: output = (Q @ cumsum(K^T @ V)) / (Q @ cumsum(K^T @ 1))
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        decay = torch.sigmoid(self.decay_param).view(1, num_heads, 1, 1)
        
        # Initialize state
        if state is None:
            state = torch.zeros(
                batch_size, num_heads, head_dim, head_dim,
                device=q.device, dtype=q.dtype
            )
        
        outputs = []
        
        # Sequential scan with decay
        for t in range(seq_len):
            q_t = q[:, :, t:t+1, :]  # [B, H, 1, D]
            k_t = k[:, :, t:t+1, :]  # [B, H, 1, D]
            v_t = v[:, :, t:t+1, :]  # [B, H, 1, D]
            
            # Update state: S = decay * S + k^T @ v
            kv = k_t.transpose(-2, -1) @ v_t  # [B, H, D, D]
            state = decay * state + kv
            
            # Compute output: o = q @ S
            o_t = q_t @ state  # [B, H, 1, D]
            
            # Normalize
            k_sum = k_t.sum(dim=-1, keepdim=True)  # [B, H, 1, 1]
            normalizer = (q_t * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            o_t = o_t / normalizer
            
            outputs.append(o_t)
        
        output = torch.cat(outputs, dim=2)  # [B, H, T, D]
        return output, state


class LocalConv(nn.Module):
    """
    Depthwise separable convolution for local patterns.
    
    Captures short-range dependencies efficiently.
    
    Args:
        hidden_size: Model dimension
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,  # Depthwise
        )
        self.pointwise = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            [batch, seq_len, hidden_size]
        """
        # Depthwise conv
        x_conv = x.transpose(1, 2)  # [B, D, T]
        x_conv = self.conv(x_conv)[:, :, :x.shape[1]]  # Causal padding
        x_conv = x_conv.transpose(1, 2)  # [B, T, D]
        
        # Pointwise projection
        return self.pointwise(F.silu(x_conv))


class UnifiedBlock(nn.Module):
    """
    Unified PULSE block combining local and global processing.
    
    Architecture:
        x → RMSNorm → [LocalConv ⊕ LinearAttn] → Gate → + → RMSNorm → FFN → + → out
                  └────────────────────────────────────┘           └─────────┘
    
    Args:
        hidden_size: Model dimension
        num_heads: Attention heads
        intermediate_size: FFN hidden dimension
        kernel_size: Local conv kernel size
        decay: Linear attention decay
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        intermediate_size: int = None,
        kernel_size: int = 4,
        decay: float = 0.95,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 2.7)
        
        # Pre-norm
        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.norm2 = RMSNorm(hidden_size, norm_eps)
        
        # Local + Global processing
        self.local_conv = LocalConv(hidden_size, kernel_size)
        self.linear_attn = LinearAttention(hidden_size, num_heads, decay)
        
        # Gated fusion
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # FFN (SwiGLU)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional recurrent state
            
        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: Updated recurrent state
        """
        # Pre-norm
        residual = x
        x_norm = self.norm1(x)
        
        # Local + Global processing
        local_out = self.local_conv(x_norm)
        global_out, new_state = self.linear_attn(x_norm, state)
        
        # Gated fusion
        combined = torch.cat([local_out, global_out], dim=-1)
        fused = torch.sigmoid(self.gate(combined)) * local_out + \
                (1 - torch.sigmoid(self.gate(combined))) * global_out
        
        # Residual
        x = residual + fused
        
        # FFN with pre-norm
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.down_proj(F.silu(self.gate_proj(x_norm)) * self.up_proj(x_norm))
        
        return x, new_state


class RecurrentState(nn.Module):
    """
    Single compressed recurrent state.
    
    Replaces complex state banks with simple EMA-style update.
    
    Args:
        hidden_size: State dimension
        momentum: EMA momentum (0.9-0.99)
    """
    
    def __init__(self, hidden_size: int, momentum: float = 0.95):
        super().__init__()
        self.hidden_size = hidden_size
        self.momentum = momentum
        
        # Learnable initial state
        self.initial_state = nn.Parameter(torch.zeros(1, hidden_size))
        
        # Projection for update
        self.update_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size * 2, 1, bias=True)
    
    def get_initial_state(self, batch_size: int) -> torch.Tensor:
        """Get initial state for batch."""
        return self.initial_state.expand(batch_size, -1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update state with new hidden states.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            state: [batch, hidden_size]
            
        Returns:
            Updated state [batch, hidden_size]
        """
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)  # [batch, hidden]
        projected = self.update_proj(pooled)
        
        # Gated update
        gate_input = torch.cat([state, projected], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # EMA-style update
        new_state = gate * state + (1 - gate) * projected
        
        return new_state
