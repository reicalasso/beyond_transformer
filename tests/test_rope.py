"""Tests for rotary positional embeddings."""

from __future__ import annotations

import torch

from pulse.modules.rope import RotaryEmbedding, apply_rope


def _make_qk(t: int, d: int = 16, h: int = 2, b: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, t, d, generator=g)
    k = torch.randn(b, h, t, d, generator=g)
    return q, k


def test_rope_preserves_norm() -> None:
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    q, k = _make_qk(8)
    cos, sin = rope.get(8, offset=0, dtype=torch.float32)
    qr, kr = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(q.pow(2).sum(-1), qr.pow(2).sum(-1), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(k.pow(2).sum(-1), kr.pow(2).sum(-1), rtol=1e-4, atol=1e-4)


def test_rope_offset_matches_full() -> None:
    """Applying RoPE in two halves with offsets must match a single full pass."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    t1, t2 = 5, 7
    q, k = _make_qk(t1 + t2)

    cos_full, sin_full = rope.get(t1 + t2, offset=0, dtype=torch.float32)
    qr_full, kr_full = apply_rope(q, k, cos_full, sin_full)

    cos1, sin1 = rope.get(t1, offset=0, dtype=torch.float32)
    cos2, sin2 = rope.get(t2, offset=t1, dtype=torch.float32)
    qr1, kr1 = apply_rope(q[:, :, :t1], k[:, :, :t1], cos1, sin1)
    qr2, kr2 = apply_rope(q[:, :, t1:], k[:, :, t1:], cos2, sin2)
    qr_split = torch.cat([qr1, qr2], dim=2)
    kr_split = torch.cat([kr1, kr2], dim=2)

    torch.testing.assert_close(qr_full, qr_split, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(kr_full, kr_split, rtol=1e-5, atol=1e-6)


def test_rope_relative_property() -> None:
    """<RoPE(q,m), RoPE(k,n)> depends only on (m-n)."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 1, 1, 16, generator=g)
    k = torch.randn(1, 1, 1, 16, generator=g)

    cos_m1, sin_m1 = rope.get(1, offset=2, dtype=torch.float32)
    cos_n1, sin_n1 = rope.get(1, offset=5, dtype=torch.float32)
    qr1, _ = apply_rope(q, q, cos_m1, sin_m1)
    _, kr1 = apply_rope(k, k, cos_n1, sin_n1)
    dot1 = (qr1 * kr1).sum()

    cos_m2, sin_m2 = rope.get(1, offset=10, dtype=torch.float32)
    cos_n2, sin_n2 = rope.get(1, offset=13, dtype=torch.float32)
    qr2, _ = apply_rope(q, q, cos_m2, sin_m2)
    _, kr2 = apply_rope(k, k, cos_n2, sin_n2)
    dot2 = (qr2 * kr2).sum()

    torch.testing.assert_close(dot1, dot2, rtol=1e-4, atol=1e-5)
