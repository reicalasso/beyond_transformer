"""
PULSE v3 — core architecture.

Single file, zero dependencies beyond torch.
No RoPE, no outer-product KV, no external memory.

Key idea: two learnable timescales per head.
  fast state  (α ≈ 0.70): local syntax, short-range patterns
  slow state  (β ≈ 0.97): semantics, long-range context

Both run as O(n) causal decay scans.
Local depthwise conv captures sub-word patterns.
Gated fusion blends all three signals.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root mean square layer normalization — no bias, no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: two parallel projections, gated by SiLU."""

    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up   = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Causal decay scan
# ---------------------------------------------------------------------------

_CHUNK = 32  # chunk size for numerically safe BF16 scan


def _decay_scan(
    x: torch.Tensor,          # [B, H, T, D]
    decay: torch.Tensor,       # [1, H, 1]   per-head scalar in (0,1)
    init: torch.Tensor,        # [B, H, D]   carry-in state
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inclusive causal prefix scan with exponential decay.

        out[t] = decay * out[t-1] + x[t],   out[-1] = init

    Returns (out [B,H,T,D], last_state [B,H,D]).

    Strategy — chunk-and-chain with closed-form cumsum:
      Within each chunk of size C the recurrence has the closed form
          out[c] = decay^c * cumsum_{s=0..c}(x[s] / decay^s) + decay^(c+1) * init
      which is computed entirely as fused CUDA ops (no per-step Python loop).
      Python iterates only over T/C chunks.  At C=32 and decay=0.95,
      decay^32 ≈ 0.19 — well above BF16 subnormals, so division is safe.
    """
    B, H, T, D = x.shape
    C = min(_CHUNK, T)

    log_d = torch.log(decay.float().clamp(min=1e-8))           # [1, H, 1]
    c_idx = torch.arange(C, device=x.device, dtype=torch.float32)

    # Power tables for one chunk: p[c] = decay^c
    lp     = log_d.unsqueeze(-1) * c_idx.view(1, 1, C, 1)     # [1,H,C,1]
    p      = lp.exp()                                           # [1,H,C,1]
    inv_p  = (-lp).exp()                                        # [1,H,C,1]
    init_p = (log_d.unsqueeze(-1) * (c_idx + 1).view(1, 1, C, 1)).exp()  # decay^(c+1)

    out   = torch.empty(B, H, T, D, device=x.device, dtype=x.dtype)
    state = init.float()  # [B, H, D] — kept in fp32 throughout

    for start in range(0, T, C):
        end  = min(start + C, T)
        clen = end - start
        chunk = x[:, :, start:end, :].float()                  # [B,H,clen,D]

        if clen < C:
            _inv_p, _p, _init_p = inv_p[:, :, :clen], p[:, :, :clen], init_p[:, :, :clen]
        else:
            _inv_p, _p, _init_p = inv_p, p, init_p

        cs         = (chunk * _inv_p).cumsum(dim=2)            # [B,H,clen,D]
        chunk_out  = cs * _p + state.unsqueeze(2) * _init_p    # [B,H,clen,D]

        out[:, :, start:end] = chunk_out.to(x.dtype)
        state = chunk_out[:, :, -1].float()                     # carry to next chunk

    return out, state


# ---------------------------------------------------------------------------
# Dual-timescale linear attention
# ---------------------------------------------------------------------------

_PHI = lambda x: F.elu(x) + 1   # non-negative kernel feature map


class DualStateAttention(nn.Module):
    """
    Linear attention with two learnable decay timescales per head.

    fast state  (α, init ≈ 0.70): captures local/syntactic patterns.
    slow state  (β, init ≈ 0.97): carries long-range semantic context.

    Both use element-wise KV accumulation — O(D) state, no outer product.

    Output:
        o[t] = W_out( gate_f * fast_out[t] + gate_s * slow_out[t] )
    where the gates are input-dependent (soft mixture of timescales).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        fast_decay: float = 0.70,
        slow_decay: float = 0.97,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.H = num_heads
        self.D = hidden_size // num_heads

        self.qkv  = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out  = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # fast+slow concat

        # Learnable decay logits — sigmoid(param) == decay at init
        def _logit(p: float) -> float:
            return math.log(p / (1.0 - p))

        self.alpha = nn.Parameter(torch.full((num_heads,), _logit(fast_decay)))
        self.beta  = nn.Parameter(torch.full((num_heads,), _logit(slow_decay)))

    def forward(
        self,
        x: torch.Tensor,                                # [B, T, H*D]
        state: Optional[Tuple] = None,                  # (kv_f, ks_f, kv_s, ks_s) each [B,H,D]
        mask: Optional[torch.Tensor] = None,            # [B, 1, 1, T] — 1=valid, 0=pad
    ) -> Tuple[torch.Tensor, Tuple]:
        B, T, _ = x.shape
        H, D = self.H, self.D

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)         # [B,H,T,D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        q = _PHI(q)
        k = _PHI(k)

        # Zero-out padding positions in k/v so they don't pollute the state
        if mask is not None:
            # mask: [B, 1, 1, T] → [B, 1, T, 1] for broadcasting over heads
            pad = mask.permute(0, 1, 3, 2)  # [B, 1, T, 1]
            k = k * pad
            v = v * pad

        kv  = k * v                                     # [B,H,T,D]  element-wise

        alpha = torch.sigmoid(self.alpha).view(1, H, 1) # [1,H,1]
        beta  = torch.sigmoid(self.beta).view(1, H, 1)

        # Unpack carry-in state
        if state is None:
            z = torch.zeros(B, H, D, device=x.device, dtype=x.dtype)
            kv_f0, ks_f0, kv_s0, ks_s0 = z, z.clone(), z.clone(), z.clone()
        else:
            kv_f0, ks_f0, kv_s0, ks_s0 = state

        # Causal scans — inclusive (token attends to itself, standard causal LM)
        run_kv_f, kv_f1 = _decay_scan(kv, alpha, kv_f0)   # [B,H,T,D]
        run_ks_f, ks_f1 = _decay_scan(k,  alpha, ks_f0)
        run_kv_s, kv_s1 = _decay_scan(kv, beta,  kv_s0)
        run_ks_s, ks_s1 = _decay_scan(k,  beta,  ks_s0)

        eps = 1e-6
        den_f = (q * run_ks_f).sum(-1, keepdim=True).clamp(min=eps)
        den_s = (q * run_ks_s).sum(-1, keepdim=True).clamp(min=eps)

        out_f = (q * run_kv_f) / den_f                 # [B,H,T,D]
        out_s = (q * run_kv_s) / den_s

        # Merge heads, concat fast+slow, project
        out_f = out_f.transpose(1, 2).contiguous().view(B, T, -1)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, -1)
        out   = self.out(torch.cat([out_f, out_s], dim=-1))

        new_state = (kv_f1, ks_f1, kv_s1, ks_s1)
        return out, new_state


# ---------------------------------------------------------------------------
# Local depthwise convolution (stateful)
# ---------------------------------------------------------------------------

class LocalConv(nn.Module):
    """
    Causal depthwise-separable convolution for sub-word / local patterns.

    padding=0 always.  First call: zero-pads left for causal alignment.
    Subsequent calls (incremental decode): prepend the saved pre-conv buffer.
    """

    def __init__(self, hidden_size: int, kernel_size: int = 4):
        super().__init__()
        self.K = kernel_size
        self.dw = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=kernel_size, padding=0, groups=hidden_size,
        )
        self.pw = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,                        # [B, T, D]
        conv_state: Optional[torch.Tensor] = None,  # [B, D, K-1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        xc = x.transpose(1, 2)                  # [B, D, T]

        if conv_state is not None:
            xc = torch.cat([conv_state, xc], dim=2)   # [B, D, K-1+T]
        else:
            pad = xc.new_zeros(xc.shape[0], xc.shape[1], self.K - 1)
            xc = torch.cat([pad, xc], dim=2)           # [B, D, K-1+T]

        new_state = xc[:, :, -(self.K - 1):].clone()  # save pre-conv buffer

        xc  = self.dw(xc)                             # [B, D, T]  exact, no trim
        xc  = xc.transpose(1, 2)                      # [B, T, D]
        return self.pw(F.silu(xc)), new_state


# ---------------------------------------------------------------------------
# Dual-state block
# ---------------------------------------------------------------------------

class DualStateBlock(nn.Module):
    """
    One transformer-style block using dual-timescale linear attention.

        x → RMSNorm → [LocalConv ⊕ DualStateAttn] → gate → residual
          → RMSNorm → SwiGLU FFN → residual

    State tuple per block: (attn_state, conv_state)
      attn_state: (kv_f, ks_f, kv_s, ks_s)  each [B, H, D]
      conv_state: [B, hidden_size, K-1]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_mult: float = 2.7,
        kernel_size: int = 4,
        fast_decay: float = 0.70,
        slow_decay: float = 0.97,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        intermediate = int(hidden_size * ffn_mult)

        self.norm1 = RMSNorm(hidden_size, norm_eps)
        self.norm2 = RMSNorm(hidden_size, norm_eps)

        self.conv   = LocalConv(hidden_size, kernel_size)
        self.attn   = DualStateAttention(hidden_size, num_heads, fast_decay, slow_decay)
        self.ffn    = SwiGLU(hidden_size, intermediate)

        # Input-dependent gate: blend conv and attention outputs
        self.gate = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        attn_state = state[0] if state is not None else None
        conv_state = state[1] if state is not None else None

        residual = x
        xn = self.norm1(x)

        conv_out, new_conv = self.conv(xn, conv_state)
        attn_out, new_attn = self.attn(xn, attn_state, mask)

        # Gated fusion of local and global signals
        combined = torch.cat([conv_out, attn_out], dim=-1)  # [B,T,2D]
        g = torch.sigmoid(self.gate(combined))               # [B,T,2D]
        fused = (g * combined).chunk(2, dim=-1)              # two [B,T,D]
        x = residual + fused[0] + fused[1]

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x, (new_attn, new_conv)