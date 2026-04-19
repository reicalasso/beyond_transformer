"""Tests for sliding-window softmax attention."""

from __future__ import annotations

import math

import pytest
import torch

from pulse.modules.swa import SlidingWindowAttention


def _module(seed: int = 0, **overrides) -> SlidingWindowAttention:
    torch.manual_seed(seed)
    cfg = dict(hidden_size=32, num_heads=4, window_size=8)
    cfg.update(overrides)
    return SlidingWindowAttention(**cfg)


def test_shape_and_finiteness() -> None:
    mod = _module()
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        out, cache = mod(x)
    assert out.shape == (2, 10, 32)
    assert torch.isfinite(out).all()
    assert cache.offset == 10


def test_window_cap() -> None:
    """Cache must never exceed window_size tokens."""
    mod = _module(window_size=4)
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        _, cache = mod(x)
    assert cache.k.shape[2] <= 4


def test_streaming_decode_matches_full() -> None:
    """Token-by-token decode equals one-shot prefill within the window."""
    mod = _module(window_size=16)
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        out_full, _ = mod(x)
        outs = []
        cache = None
        for t in range(x.shape[1]):
            o, cache = mod(x[:, t : t + 1], cache=cache)
            outs.append(o)
        out_stream = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_full, out_stream, rtol=1e-4, atol=1e-4)


def test_causality_no_future_leak() -> None:
    """Changing future tokens must not change past outputs."""
    mod = _module(window_size=16)
    x1 = torch.randn(1, 8, 32)
    x2 = x1.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)

    with torch.no_grad():
        out1, _ = mod(x1)
        out2, _ = mod(x2)

    torch.testing.assert_close(out1[:, :5], out2[:, :5], rtol=1e-4, atol=1e-5)


def test_gqa_runs() -> None:
    mod = _module(num_heads=4, num_kv_heads=2)
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        out, _ = mod(x)
    assert out.shape == (1, 6, 32)


def test_padding_mask_zeros_attention_to_pad() -> None:
    """Tokens at positions marked invalid must not contribute to softmax."""
    mod = _module(window_size=16)
    x = torch.randn(1, 5, 32)
    mask = torch.tensor([[1, 1, 0, 1, 1]])  # mark token 2 as pad

    x_no_pad = x.clone()
    x_no_pad[:, 2] = torch.randn_like(x[:, 2]) * 1e6  # huge garbage at pad slot

    with torch.no_grad():
        out_clean, _ = mod(x, attention_mask=mask)
        out_garbage, _ = mod(x_no_pad, attention_mask=mask)

    torch.testing.assert_close(
        out_clean[:, [0, 1, 3, 4]],
        out_garbage[:, [0, 1, 3, 4]],
        rtol=1e-4,
        atol=1e-4,
    )


def test_gradients_flow() -> None:
    mod = _module()
    x = torch.randn(1, 6, 32, requires_grad=True)
    out, _ = mod(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
