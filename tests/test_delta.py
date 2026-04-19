"""Tests for the gated delta-rule recurrence.

These are the load-bearing correctness tests for the new architecture:
* Recurrent and chunked paths must produce identical outputs.
* Chunk-size invariance for any divisor.
* Per-token streaming decode must match full-sequence prefill.
* Forward gradients must be finite under autograd.
* QK-norm and gate biases must produce reasonable initial outputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from pulse.modules.delta import DeltaState, GatedDeltaRule


def _module(seed: int = 0, **overrides) -> GatedDeltaRule:
    torch.manual_seed(seed)
    cfg = dict(hidden_size=32, num_heads=4, qk_norm=True, chunk_size=8)
    cfg.update(overrides)
    return GatedDeltaRule(**cfg)


def test_recurrent_and_chunked_match() -> None:
    mod = _module()
    x = torch.randn(2, 24, 32)

    with torch.no_grad():
        out_rec, st_rec = mod(x, use_chunked=False)
        out_chk, st_chk = mod(x, use_chunked=True)

    torch.testing.assert_close(out_rec, out_chk, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(st_rec.S, st_chk.S, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 16, 24])
def test_chunk_size_invariance(chunk: int) -> None:
    mod = _module(chunk_size=4)
    x = torch.randn(1, 24, 32)

    with torch.no_grad():
        out_ref, st_ref = mod(x, use_chunked=False)
        mod.chunk_size = chunk
        out_var, st_var = mod(x, use_chunked=True)

    torch.testing.assert_close(out_ref, out_var, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(st_ref.S, st_var.S, rtol=1e-5, atol=1e-5)


def test_streaming_decode_matches_prefill() -> None:
    """Prefill all-at-once vs token-by-token incremental decode."""
    mod = _module()
    x = torch.randn(1, 16, 32)

    with torch.no_grad():
        out_full, st_full = mod(x, use_chunked=True)

        outs = []
        st = None
        for t in range(x.shape[1]):
            o, st = mod(x[:, t : t + 1], state=st)
            outs.append(o)
        out_stream = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_full, out_stream, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(st_full.S, st.S, rtol=1e-4, atol=1e-5)


def test_split_prefill_then_decode() -> None:
    """Half prefill, then per-token decode for the rest."""
    mod = _module()
    x = torch.randn(1, 12, 32)

    with torch.no_grad():
        out_full, st_full = mod(x, use_chunked=True)

        out_a, st_a = mod(x[:, :7], use_chunked=True)
        outs = [out_a]
        st = st_a
        for t in range(7, 12):
            o, st = mod(x[:, t : t + 1], state=st)
            outs.append(o)
        out_split = torch.cat(outs, dim=1)

    torch.testing.assert_close(out_full, out_split, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(st_full.S, st.S, rtol=1e-4, atol=1e-5)


def test_gradients_flow_and_are_finite() -> None:
    mod = _module()
    x = torch.randn(2, 8, 32, requires_grad=True)
    out, _ = mod(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for p in mod.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_state_shape_and_dtype() -> None:
    mod = _module()
    state = mod.empty_state(3, torch.device("cpu"), torch.float32)
    assert isinstance(state, DeltaState)
    assert state.S.shape == (3, mod.num_heads, mod.head_dim, mod.head_dim)
    assert state.S.dtype == torch.float32


def test_zero_input_zero_output() -> None:
    """With zero inputs and the bias-only gates the output should be near zero."""
    mod = _module()
    x = torch.zeros(2, 5, 32)
    with torch.no_grad():
        out, st = mod(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    assert torch.allclose(st.S, torch.zeros_like(st.S), atol=1e-6)


def test_alpha_init_high_retention() -> None:
    """With gate_bias_init high, sigmoid(α) should be close to 1 at init."""
    mod = _module(gate_bias_init=4.0)
    x = torch.zeros(1, 1, 32)
    alpha_logit = mod.alpha_proj(x)
    alpha = torch.sigmoid(alpha_logit)
    assert (alpha > 0.95).all(), f"alpha at init too low: {alpha.min().item()}"


def test_no_qk_norm_path() -> None:
    """Module must also work with qk_norm disabled."""
    mod = _module(qk_norm=False)
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        out, _ = mod(x)
    assert torch.isfinite(out).all()


def test_repeat_call_changes_state() -> None:
    """Carrying state across two calls should not be a no-op."""
    mod = _module()
    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        _, st1 = mod(x)
        _, st2 = mod(x, state=st1)
    assert not torch.allclose(st1.S, st2.S)
