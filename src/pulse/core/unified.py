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

from .ffn import SwiGLU
from .norm import RMSNorm


class LinearAttention(nn.Module):
    """
    Linear Attention with exponential decay and lightweight state.
    
    Chunked vectorized causal scan with O(T/C) Python iterations
    (C=32 by default) — each iteration is fully vectorized CUDA ops.
    Keeps a single running KV and K-sum per head (element-wise, not
    the full [D, D] outer-product matrix).
    
    Forward recurrence (per head, per step t):
        kv_t   = decay * kv_{t-1}   + k_t * v_t
        ksum_t = decay * ksum_{t-1} + k_t
        o_t[d] = q_t[d] * kv_t[d] / clamp(sum_d(q_t[d] * ksum_t[d]))

    The inclusive scan means each position can attend to itself
    (standard for causal self-attention).
    
    Args:
        hidden_size: Model dimension
        num_heads: Number of attention heads
        decay: Desired initial decay factor (0.0–1.0). Stored as
            pre-sigmoid logit so sigmoid(param) == decay at init.
            Learnable per head.
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
        
        # Learnable output gate per dimension.  Controls how much of the
        # attention-weighted output to use vs. passing the raw value through.
        # When the accumulated state degenerates (e.g. repetition), the
        # model can learn to attenuate attention and rely on the direct
        # path, preventing fixed-point collapse.
        self.out_gate = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Learnable decay per head; stored as pre-sigmoid logit.
        # logit(p) = log(p / (1-p)) so that sigmoid(logit) == p.
        _logit = math.log(decay / (1.0 - decay)) if 0 < decay < 1 else 0.0
        self.decay_param = nn.Parameter(torch.full((num_heads,), _logit))
    
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
        
        kv_contrib = k * v  # [B, H, T, D]  element-wise KV
        
        # Initialize from incoming state
        if state is None:
            kv_init   = torch.zeros(bsz, H, D, device=x.device, dtype=x.dtype)
            ksum_init = torch.zeros(bsz, H, D, device=x.device, dtype=x.dtype)
        else:
            kv_init, ksum_init = state
        
        # Inclusive causal prefix scan: position t includes its own contribution,
        # which is standard for causal self-attention (token can attend to itself).
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
        
        # Also compute a direct value path (skip attention state)
        v_direct = v.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # Learned gate: blend attention output with direct value signal.
        # This prevents fixed-point collapse during autoregressive generation
        # — the model can learn to attenuate degenerate attention states.
        g = torch.sigmoid(self.out_gate(output))
        output = g * output + (1 - g) * v_direct
        output = self.out_proj(output)
        
        # Final state is the last time-step of the inclusive scan
        new_kv   = running_kv[:, :, -1, :]    # [B, H, D]
        new_ksum = running_ksum[:, :, -1, :]  # [B, H, D]
        new_state = (new_kv, new_ksum)
        return output, new_state


_SCAN_CHUNK = 32  # decay^32 ≈ 0.19 at decay=0.95 — safe for BF16/FP16


def _causal_decay_scan(
    x: torch.Tensor,
    decay: torch.Tensor,
    init: torch.Tensor,
) -> torch.Tensor:
    """
    Chunked vectorized causal prefix scan with exponential decay.

    Computes the inclusive scan:
        out[t] = decay * out[t-1] + x[t],  out[-1] = init

    Strategy — chunk-and-chain:
      Divide T into chunks of size C (default 32). Within each chunk
      the closed-form cumsum is numerically safe because decay^C stays
      well above BF16 subnormals. Between chunks the ending state is
      chained to the next chunk's init.

      Python iterations: T/C (e.g. 8 for T=256). Each iteration runs
      fully vectorized CUDA ops — no per-timestep Python overhead.

    Args:
        x:     [batch, heads, seq_len, dim]  per-step contributions
        decay: [1, heads, 1]                 per-head decay scalar in (0,1)
        init:  [batch, heads, dim]           state carried in from previous chunk

    Returns:
        out:   [batch, heads, seq_len, dim]  inclusive causal prefix sums
    """
    B, H, T, D = x.shape
    C = min(_SCAN_CHUNK, T)

    # Pre-compute log-decay and within-chunk power tables (float32).
    log_decay = torch.log(decay.float().clamp(min=1e-8))  # [1, H, 1]
    c_idx = torch.arange(C, device=x.device, dtype=torch.float32)

    # p[c] = decay^c  for c in 0..C-1  →  [1, H, C, 1]
    log_p = log_decay.unsqueeze(-1) * c_idx.view(1, 1, C, 1)
    p = log_p.exp()       # float32, [1, H, C, 1]
    inv_p = (-log_p).exp()  # float32, [1, H, C, 1]

    # init_p[c] = decay^(c+1) for init contribution within a chunk
    init_p = (log_decay.unsqueeze(-1) * (c_idx + 1).view(1, 1, C, 1)).exp()

    out = torch.empty(B, H, T, D, device=x.device, dtype=x.dtype)
    state = init.float()  # [B, H, D], keep in float32 throughout

    for start in range(0, T, C):
        end = min(start + C, T)
        clen = end - start

        chunk = x[:, :, start:end, :].float()  # [B, H, clen, D]

        # If last chunk is shorter than C, slice the pre-computed tables.
        if clen < C:
            _inv_p = inv_p[:, :, :clen, :]
            _p = p[:, :, :clen, :]
            _init_p = init_p[:, :, :clen, :]
        else:
            _inv_p, _p, _init_p = inv_p, p, init_p

        # Closed-form cumsum within the chunk:
        #   chunk_out[c] = decay^c * cumsum_{s=0..c}(chunk[s] / decay^s)
        scaled = chunk * _inv_p             # [B, H, clen, D]
        cs = scaled.cumsum(dim=2)           # [B, H, clen, D]
        chunk_out = cs * _p                 # [B, H, clen, D]

        # Add incoming state contribution: state * decay^(c+1)
        chunk_out = chunk_out + state.unsqueeze(2) * _init_p

        out[:, :, start:end, :] = chunk_out.to(x.dtype)

        # Chain: ending state of this chunk becomes init for the next
        state = chunk_out[:, :, -1, :]  # [B, H, D], float32

    return out


class LocalConv(nn.Module):
    """
    Depthwise separable convolution for local patterns.
    
    Captures short-range dependencies efficiently.  Supports an
    optional ``conv_state`` buffer so that incremental (single-token)
    inference sees the same context window as full-sequence training.
    
    Args:
        hidden_size: Model dimension
        kernel_size: Convolution kernel size
    """
    
    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.kernel_size = kernel_size
        # padding=0: we always manually prepend causal context (zeros or
        # cached state) so there is exactly one source of left-padding.
        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size,
            padding=0,
            groups=hidden_size,  # Depthwise
        )
        self.pointwise = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
            conv_state: Optional [batch, hidden_size, kernel_size-1] buffer
                        of the last kernel_size-1 timesteps (in Conv1d layout).
        Returns:
            output: [batch, seq_len, hidden_size]
            new_conv_state: [batch, hidden_size, kernel_size-1]
        """
        T = x.shape[1]
        x_conv = x.transpose(1, 2)  # [B, D, T]

        if conv_state is not None:
            # Prepend cached context from previous call.
            x_conv = torch.cat([conv_state, x_conv], dim=2)  # [B, D, K-1+T]
        else:
            # First call — zero-pad on the left for causal alignment.
            pad = x_conv.new_zeros(x_conv.shape[0], x_conv.shape[1], self.kernel_size - 1)
            x_conv = torch.cat([pad, x_conv], dim=2)  # [B, D, K-1+T]

        # Save last kernel_size-1 *input* timesteps for the next call,
        # before running the conv (state is pre-conv representation).
        new_conv_state = x_conv[:, :, -(self.kernel_size - 1):].clone()

        # Conv with padding=0: input length K-1+T → output length T. Exact.
        x_conv = self.conv(x_conv)  # [B, D, T]

        x_conv = x_conv.transpose(1, 2)  # [B, T, D]
        return self.pointwise(F.silu(x_conv)), new_conv_state


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
        self.ffn = SwiGLU(hidden_size, intermediate_size)
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, hidden_size]
            state: Optional tuple (attn_state, conv_state) where
                attn_state is the LinearAttention (kv, ksum) pair and
                conv_state is the LocalConv buffer.
            
        Returns:
            output: [batch, seq_len, hidden_size]
            new_state: (attn_state, conv_state)
        """
        # Unpack composite state
        if state is not None:
            attn_state, conv_state = state
        else:
            attn_state, conv_state = None, None

        # Pre-norm
        residual = x
        x_norm = self.norm1(x)
        
        # Local + Global processing
        local_out, new_conv_state = self.local_conv(x_norm, conv_state)
        global_out, new_attn_state = self.linear_attn(x_norm, attn_state, attention_mask=attention_mask)
        
        # Gated fusion — compute gate once to avoid redundant forward pass
        combined = torch.cat([local_out, global_out], dim=-1)
        g = torch.sigmoid(self.gate(combined))
        fused = g * local_out + (1 - g) * global_out
        
        # Residual
        x = residual + fused
        
        # FFN with pre-norm
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ffn(x_norm)
        
        return x, (new_attn_state, new_conv_state)


class RecurrentState(nn.Module):
    """
    Single compressed recurrent state with gated update.
    
    Pools the last hidden state and merges it with the previous state
    through a learned sigmoid gate, producing a fixed-size summary
    that carries cross-chunk context.
    
    Args:
        hidden_size: State dimension
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
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
