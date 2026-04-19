"""Sliding-window softmax attention with RoPE and KV cache.

Used as the "recall" sub-layer in PULSE's hybrid stack: every Nth block runs
exact softmax attention over the last ``window_size`` tokens. This restores
exact in-context retrieval without the quadratic blow-up of full attention.

Architectural choices:
  * RoPE applied to (q, k) only inside the window.
  * Optional QK-norm for stability at scale.
  * GQA-style head sharing supported via ``num_kv_heads`` (defaults to
    ``num_heads`` = standard MHA).
  * KV cache is a fixed-capacity ring buffer of size ``window_size`` for
    O(window_size) per-token decoding memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import l2_normalize
from .rope import RotaryEmbedding, apply_rope


@dataclass
class SWACache:
    """Sliding-window KV cache.

    Attributes:
        k: Cached keys, shape ``[B, H_kv, L, D]`` where ``L <= window_size``.
        v: Cached values, same shape as ``k``.
        offset: Total number of tokens consumed (used for RoPE position).
    """

    k: torch.Tensor
    v: torch.Tensor
    offset: int


class SlidingWindowAttention(nn.Module):
    """Causal sliding-window softmax attention.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of query heads.
        window_size: Maximum attention window length (tokens).
        num_kv_heads: Number of key/value heads. ``None`` → equals
            ``num_heads`` (MHA). Smaller value → GQA.
        qk_norm: Apply L2 normalization on q, k.
        rope_base: RoPE frequency base.
        rope_max_seq_len: Initial RoPE cache length.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        num_kv_heads: int | None = None,
        qk_norm: bool = True,
        rope_base: float = 10_000.0,
        rope_max_seq_len: int = 4096,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            max_seq_len=rope_max_seq_len,
            base=rope_base,
        )

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> SWACache:
        return SWACache(
            k=torch.zeros(
                batch_size, self.num_kv_heads, 0, self.head_dim, device=device, dtype=dtype
            ),
            v=torch.zeros(
                batch_size, self.num_kv_heads, 0, self.head_dim, device=device, dtype=dtype
            ),
            offset=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: SWACache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SWACache]:
        """Forward.

        Args:
            x: ``[B, T, hidden_size]``.
            cache: Optional KV cache from a previous segment.
            attention_mask: ``[B, T]`` with 1 = valid, 0 = pad. Pad positions
                are masked from being attended to.

        Returns:
            ``(output, new_cache)``.
        """
        b, t, _ = x.shape
        h, hk, d = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(b, t, hk, d).transpose(1, 2)  # [B, Hk, T, D]
        v = self.v_proj(x).view(b, t, hk, d).transpose(1, 2)  # [B, Hk, T, D]

        if self.qk_norm:
            q = l2_normalize(q, dim=-1)
            k = l2_normalize(k, dim=-1)

        offset = cache.offset if cache is not None else 0
        cos, sin = self.rope.get(t, offset=offset, device=x.device, dtype=torch.float32)
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None and cache.k.shape[2] > 0:
            k_full = torch.cat([cache.k, k], dim=2)
            v_full = torch.cat([cache.v, v], dim=2)
        else:
            k_full = k
            v_full = v

        # NOTE: do NOT truncate k_full here. The mask below enforces both
        # causality and the sliding window; truncating before masking would
        # silently drop keys that earlier queries are still allowed to see.
        # We only cap what gets *saved* into the next-step cache.
        kv_len = k_full.shape[2]
        new_cache = SWACache(
            k=k_full[:, :, -self.window_size :].detach(),
            v=v_full[:, :, -self.window_size :].detach(),
            offset=offset + t,
        )

        if hk != h:
            repeat = h // hk
            k_full = k_full.repeat_interleave(repeat, dim=1)
            v_full = v_full.repeat_interleave(repeat, dim=1)

        q_pos = torch.arange(kv_len - t, kv_len, device=x.device)  # [T]
        k_pos = torch.arange(kv_len, device=x.device)  # [kv_len]
        causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [T, kv_len]
        window = k_pos.unsqueeze(0) > (q_pos.unsqueeze(1) - self.window_size)
        allowed = causal & window  # [T, kv_len]

        if attention_mask is not None:
            pad_full = attention_mask
            if cache is not None and cache.k.shape[2] > 0:
                pad_prev = pad_full.new_ones(b, cache.k.shape[2])
                pad_full = torch.cat([pad_prev, attention_mask], dim=1)
            if pad_full.shape[1] > self.window_size:
                pad_full = pad_full[:, -self.window_size :]
            pad_mask = pad_full.bool().unsqueeze(1).unsqueeze(1)  # [B, 1, 1, kv_len]
        else:
            pad_mask = None

        attn_logits = torch.matmul(q, k_full.transpose(-1, -2)) * self.scale  # [B, H, T, kv_len]
        mask = allowed.unsqueeze(0).unsqueeze(0)
        if pad_mask is not None:
            mask = mask & pad_mask
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1).to(q.dtype)
        out = torch.matmul(attn, v_full)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(b, t, h * d)
        return self.out_proj(out), new_cache
