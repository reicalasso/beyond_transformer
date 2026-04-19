"""Rotary positional embeddings (RoPE).

Standard half-rotation formulation, applied to the head-dim of (q, k) tensors
shaped ``[B, H, T, D]`` (D must be even). Frequencies are cached on the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Precompute and cache (cos, sin) tables for RoPE.

    Args:
        head_dim: Per-head channel dimension (must be even).
        max_seq_len: Initial cache size; auto-grows on demand.
        base: Frequency base (10_000 in the original RoPE paper).
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.inv_freq: torch.Tensor
        self._cos_cache: torch.Tensor
        self._sin_cache: torch.Tensor
        self._build_cache(max_seq_len, device=inv_freq.device, dtype=torch.float32)

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [T, D/2]
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        self.register_buffer("_cos_cache", cos, persistent=False)
        self.register_buffer("_sin_cache", sin, persistent=False)
        self._cached_len = seq_len

    def get(
        self,
        seq_len: int,
        offset: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape ``[seq_len, head_dim/2]`` starting at ``offset``."""
        end = offset + seq_len
        target_device = device or self._cos_cache.device
        if (
            end > self._cached_len
            or self._cos_cache.device != target_device
            or self._cos_cache.dtype != dtype
        ):
            new_len = max(end, self._cached_len * 2)
            self._build_cache(new_len, device=target_device, dtype=dtype)
        cos = self._cos_cache[offset:end]
        sin = self._sin_cache[offset:end]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split last dim in halves and rotate: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to ``q`` and ``k`` of shape ``[B, H, T, D]``.

    ``cos``/``sin`` arrive shaped ``[T, D/2]`` and are duplicated to ``[T, D]``
    so the rotation matches the half-split convention of ``_rotate_half``.
    """
    cos_full = torch.cat([cos, cos], dim=-1)  # [T, D]
    sin_full = torch.cat([sin, sin], dim=-1)
    cos_b = cos_full.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
    sin_b = sin_full.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)
