"""
PULSE Unified Block

Single primitive combining:
- Local patterns: Depthwise convolution
- Global context: Linear attention with decay
- Gated fusion

O(n) complexity, replaces separate SSM/Attention/State modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class LinearAttention(nn.Module):
    """
    Linear Attention with exponential decay and lightweight state.
    
    Implementation follows kernel-based causal attention in a fully
    vectorized way and keeps only a single running KV and K sum per
    head instead of a full [D, D] matrix.
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        decay: Exponential decay factor (0.0-1.0). 0.0 disables decay.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        decay: float = 0.95,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Learnable decay per head in [0, 1]
        self.decay_param = nn.Parameter(torch.full((num_heads,), float(decay)))
    
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel feature map: elu(x) + 1 to keep values non-negative."""
        return F.elu(x) + 1
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional recurrent state.
        
        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional tuple (kv, k_sum) where
                kv: [batch, num_heads, head_dim] running KV summary
                k_sum: [batch, num_heads, head_dim] running key sum
            attention_mask: Optional [batch, 1, 1, seq_len] mask where 1
                marks valid tokens and 0 marks padding positions. When
                provided, padded positions do not contribute to the
                running state.
            
        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: (kv, k_sum) with same shapes as above
        """
        bsz, seqlen, _ = x.shape
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self._phi(q)
        k = self._phi(k)
        
        # Initialize state
        if state is None:
            kv = torch.zeros(
                bsz, self.num_heads, self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            k_sum = torch.zeros_like(kv)
        else:
            kv, k_sum = state
        
        # Expand decay to broadcast over sequence
        decay = torch.sigmoid(self.decay_param).view(1, self.num_heads, 1)
        
        # Prepare boolean mask over time if provided: [B, 1, 1, T] -> [B, 1, T]
        if attention_mask is not None:
            # 1.0 for valid tokens, 0.0 for padding
            time_mask = attention_mask.squeeze(2)  # [B, 1, T]
        else:
            time_mask = None
        
        # We perform a causal scan along time dimension using
        # cumulative updates of kv and k_sum with decay.
        outputs = []
        for t in range(seqlen):
            k_t = k[:, :, t, :]  # [B, H, D]
            v_t = v[:, :, t, :]  # [B, H, D]
            q_t = q[:, :, t, :]  # [B, H, D]
            
            if time_mask is not None:
                m_t = time_mask[:, :, t]  # [B, 1]
                m_t = m_t.expand(-1, self.num_heads)  # [B, H]
                m_t = m_t.unsqueeze(-1)  # [B, H, 1]
                k_t = k_t * m_t
                v_t = v_t * m_t
            
            # Update running statistics
            kv = decay * kv + torch.bmm(
                k_t.view(bsz * self.num_heads, 1, self.head_dim),
                v_t.view(bsz * self.num_heads, self.head_dim, 1),
            ).view(bsz, self.num_heads, self.head_dim)
            k_sum = decay * k_sum + k_t
            
            # Compute attention output
            num = (q_t * kv).sum(dim=-1, keepdim=True)  # [B, H, 1]
            den = (q_t * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            o_t = (num / den) * q_t  # [B, H, D]
            outputs.append(o_t.unsqueeze(2))
        
        output = torch.cat(outputs, dim=2)  # [B, H, T, D]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.out_proj(output)
        
        new_state = (kv, k_sum)
        return output, new_state


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
        attention_mask: torch.Tensor = None,
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
        global_out, new_state = self.linear_attn(x_norm, state, attention_mask=attention_mask)
        
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
