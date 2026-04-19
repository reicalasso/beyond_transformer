"""Tests for the streaming short causal conv."""

from __future__ import annotations

import torch

from pulse.modules.conv import ShortCausalConv1d


def test_conv_shape() -> None:
    conv = ShortCausalConv1d(channels=16, kernel_size=4)
    x = torch.randn(2, 10, 16)
    y, state = conv(x)
    assert y.shape == (2, 10, 16)
    assert state.shape == (2, 16, 3)


def test_conv_streaming_matches_full() -> None:
    """Splitting the sequence and threading state must equal the full pass."""
    torch.manual_seed(0)
    conv = ShortCausalConv1d(channels=8, kernel_size=4)
    x = torch.randn(1, 12, 8)

    with torch.no_grad():
        y_full, _ = conv(x)
        y1, st1 = conv(x[:, :5])
        y2, st2 = conv(x[:, 5:9], state=st1)
        y3, _ = conv(x[:, 9:], state=st2)

    y_stream = torch.cat([y1, y2, y3], dim=1)
    torch.testing.assert_close(y_full, y_stream, rtol=1e-5, atol=1e-6)


def test_conv_kernel_size_1_is_pointwise() -> None:
    conv = ShortCausalConv1d(channels=4, kernel_size=1, activation=None, bias=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
    x = torch.randn(1, 3, 4)
    y, _ = conv(x)
    torch.testing.assert_close(y, x, rtol=1e-6, atol=1e-6)
