"""Tests for normalization primitives."""

from __future__ import annotations

import torch

from pulse.modules.norm import RMSNorm, l2_normalize


def test_rmsnorm_unit_variance() -> None:
    norm = RMSNorm(64)
    x = torch.randn(4, 16, 64) * 5.0
    y = norm(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_weight_scales_output() -> None:
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(2, 3, 8)
    y = norm(x)
    rms = y.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-3)


def test_l2_normalize_unit_norm() -> None:
    x = torch.randn(2, 4, 8, 16)
    y = l2_normalize(x, dim=-1)
    norms = y.float().pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_l2_normalize_handles_zero() -> None:
    x = torch.zeros(2, 4)
    y = l2_normalize(x, dim=-1)
    assert torch.isfinite(y).all()
