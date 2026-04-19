"""Tests for hybrid blocks (DeltaBlock, AttentionBlock)."""

from __future__ import annotations

import torch

from pulse.modules.block import AttentionBlock, BlockState, DeltaBlock


def test_delta_block_forward_and_streaming() -> None:
    torch.manual_seed(0)
    block = DeltaBlock(hidden_size=32, num_heads=4, conv_kernel_size=3, chunk_size=4)
    x = torch.randn(1, 12, 32)

    with torch.no_grad():
        out_full, st_full = block(x)
        outs = []
        st = None
        for t in range(x.shape[1]):
            o, st = block(x[:, t : t + 1], state=st)
            outs.append(o)
        out_stream = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_full, out_stream, rtol=1e-3, atol=1e-4)
    assert st.conv_state.shape == st_full.conv_state.shape


def test_attention_block_forward_and_streaming() -> None:
    torch.manual_seed(0)
    block = AttentionBlock(hidden_size=32, num_heads=4, window_size=16, conv_kernel_size=3)
    x = torch.randn(1, 8, 32)

    with torch.no_grad():
        out_full, st_full = block(x)
        outs = []
        st = None
        for t in range(x.shape[1]):
            o, st = block(x[:, t : t + 1], state=st)
            outs.append(o)
        out_stream = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_full, out_stream, rtol=1e-3, atol=1e-4)


def test_block_state_detach() -> None:
    block = DeltaBlock(hidden_size=16, num_heads=2)
    x = torch.randn(1, 4, 16, requires_grad=True)
    _, st = block(x)
    detached = st.detach()
    assert not detached.conv_state.requires_grad
