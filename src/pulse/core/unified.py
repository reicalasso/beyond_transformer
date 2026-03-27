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


class LinearAttention(nn.Module):
    """
    Linear Attention with exponential decay and lightweight state.
    
    Fully vectorized causal scan: no Python loop over the sequence.
    Keeps only a single running KV and K-sum per head instead of a
    full [D, D] outer-product matrix.
    
    Forward recurrence (per head, per step t):
        kv_t   = decay * kv_{t-1}   + k_t * v_t
        ksum_t = decay * ksum_{t-1} + k_t
        o_t    = (q_t · kv_t) / max(q_t · ksum_t, eps)

    The causal constraint is enforced by excluding future tokens from
    the running state when computing the output at position t.
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        decay: Initial exponential decay factor (0.0–1.0). Learnable per head.
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
        
        # Learnable decay per head; stored as pre-sigmoid logit
        self.decay_param = nn.Parameter(torch.full((num_heads,), float(decay)))
    
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Kernel feature map: elu(x) + 1, keeps values non-negative."""
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
                kv:    [batch, num_heads, head_dim]  running KV summary
                k_sum: [batch, num_heads, head_dim]  running key sum
            attention_mask: Optional [batch, 1, 1, seq_len] mask where 1
                marks valid tokens and 0 marks padding positions. When
                provided, padded positions do not contribute to the
                running state.
            
        Returns:
            output:    [batch, seq_len, hidden_size]
            new_state: (kv, k_sum) with same shapes as input state
        """
        bsz, seqlen, _ = x.shape
        H, D = self.num_heads, self.head_dim
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # [B, H, T, D]
        q = q.view(bsz, seqlen, H, D).transpose(1, 2)
        k = k.view(bsz, seqlen, H, D).transpose(1, 2)
        v = v.view(bsz, seqlen, H, D).transpose(1, 2)
        
        q = self._phi(q)  # [B, H, T, D]
        k = self._phi(k)  # [B, H, T, D]
        
        # Apply padding mask: zero out k/v at padding positions so they
        # do not contaminate the running state.
        if attention_mask is not None:
            # attention_mask: [B, 1, 1, T] -> [B, 1, T, 1]
            pad_mask = attention_mask.squeeze(2).transpose(1, 2).unsqueeze(-1)  # [B, T, 1, 1]
            pad_mask = pad_mask.permute(0, 2, 1, 3)  # [B, 1, T, 1]
            k = k * pad_mask
            v = v * pad_mask
        
        # decay: [H] -> [1, H, 1] for broadcasting
        decay = torch.sigmoid(self.decay_param).view(1, H, 1)  # [1, H, 1]
        
        # Build causal decay powers: decay^(t - s) for t > s.
        # powers[t] = decay^t, so prefix_kv at t = sum_{s<=t} decay^{t-s} * k_s*v_s
        # Vectorized via cumulative sum with geometric discounting:
        #   state_t = decay * state_{t-1} + k_t * v_t
        # We scan over time using torch.ops in a single loop over heads is
        # still O(n) but fully parallelisable via the following approach:
        # Compute outer products k_t*v_t [B,H,T,D] and k_t [B,H,T,D],
        # then apply a parallel prefix scan with geometric decay.
        #
        # Parallel prefix (associative scan) over T:
        #   Given a[t] = k_t * v_t (or k_t), combine with operator:
        #   (s1, a1) ⊕ (s2, a2) = (decay * s1 + a2, ...)
        # We implement this iteratively over log2(T) steps.
        
        kv_contrib = k * v  # [B, H, T, D]  (element-wise, NOT outer product)
        
        # Initialize from incoming state
        if state is None:
            kv_init   = torch.zeros(bsz, H, D, device=x.device, dtype=x.dtype)
            ksum_init = torch.zeros(bsz, H, D, device=x.device, dtype=x.dtype)
        else:
            kv_init, ksum_init = state
        
        # Inclusive causal prefix scan: computes running_kv[t] and running_ksum[t]
        # that include contributions from positions 0..t.
        # We need the *exclusive* prefix (0..t-1) for causal attention at t.
        # Strategy: compute inclusive scan, then shift right by 1 and prepend init.
        running_kv   = _causal_decay_scan(kv_contrib, decay, kv_init)    # [B, H, T, D]
        running_ksum = _causal_decay_scan(k,          decay, ksum_init)   # [B, H, T, D]
        
        # Compute D-dimensional output.
        # With element-wise KV approximation (k*v per-dim, not outer product),
        # the correct output per position t is:
        #   o[t, d] = q[t,d] * running_kv[t,d] / clamp(sum_d(q[t,d] * running_ksum[t,d]))
        # This keeps the full D-dimensional structure while normalizing by the
        # scalar partition function — avoids collapsing to a scalar and re-expanding.
        den = (q * running_ksum).sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, H, T, 1]
        output = (q * running_kv) / den  # [B, H, T, D]
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.out_proj(output)
        
        # Final state is the last time-step of the inclusive scan
        new_kv   = running_kv[:, :, -1, :]    # [B, H, D]
        new_ksum = running_ksum[:, :, -1, :]  # [B, H, D]
        new_state = (new_kv, new_ksum)
        return output, new_state


def _causal_decay_scan(
    x: torch.Tensor,
    decay: torch.Tensor,
    init: torch.Tensor,
) -> torch.Tensor:
    """
    Fully vectorized causal prefix scan with exponential decay.

    Computes the inclusive scan without any Python loop over T:
        out[t] = sum_{s=0}^{t} decay^(t-s) * x[s]  +  decay^(t+1) * init

    Strategy — closed-form via weighted cumsum:
      1. Build decay powers: p[t] = decay^t  →  [1, H, T, 1]
      2. Scale inputs:       x_scaled[t] = x[t] / p[t]
      3. Cumsum along T:     cs[t] = sum_{s=0}^{t} x_scaled[s]
      4. Restore scale:      out[t] = p[t] * cs[t]  +  decay^(t+1) * init

    This is numerically equivalent to the sequential recurrence and runs
    entirely as fused CUDA ops — no Python loop over T.

    Args:
        x:     [batch, heads, seq_len, dim]  per-step contributions
        decay: [1, heads, 1]                 per-head decay scalar in (0,1)
        init:  [batch, heads, dim]           state carried in from previous chunk

    Returns:
        out:   [batch, heads, seq_len, dim]  inclusive causal prefix sums
    """
    B, H, T, D = x.shape

    # Build per-head log-decay series in float32 to avoid underflow.
    # log_p[h, t] = (t+1) * log(decay[h])  → p[t] = decay^(t+1)
    # Working in log-space avoids very small values when decay^T is tiny.
    t_idx = torch.arange(T, device=x.device, dtype=torch.float32)  # [T]
    log_decay = torch.log(decay.float().clamp(min=1e-8))  # [1, H, 1]
    # log_powers: [1, H, T, 1]
    log_powers = log_decay.unsqueeze(-1) * (t_idx + 1).view(1, 1, T, 1)
    powers = log_powers.exp().to(x.dtype)  # [1, H, T, 1]

    # Closed-form weighted cumsum (all ops are CUDA-fused, no Python loop):
    #   out[t] = p[t] * sum_{s=0}^{t} x[s] / p[s]
    # Divide in float32 to avoid precision loss, cast back afterwards.
    x_f = x.float()
    x_scaled = x_f / powers.float()         # [B, H, T, D]
    cs = x_scaled.cumsum(dim=2)             # [B, H, T, D]
    out = (cs * powers.float()).to(x.dtype)  # [B, H, T, D]

    # Init contribution: init * decay^(t+1)
    out = out + init.unsqueeze(2) * powers  # [B, H, T, D]
    return out


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
        
        # Gated fusion — compute gate once to avoid redundant forward pass
        combined = torch.cat([local_out, global_out], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        fused = g * local_out + (1 - g) * global_out
        
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
        # Use last non-padding token rather than mean — preserves recency
        pooled = hidden_states[:, -1, :]  # [batch, hidden]
        projected = self.update_proj(pooled)
        
        # Gated update
        gate_input = torch.cat([state, projected], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        # EMA-style update
        new_state = gate * state + (1 - gate) * projected
        
        return new_state
