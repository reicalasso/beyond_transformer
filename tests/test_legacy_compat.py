"""Sanity check: the legacy v3 prototype is still importable and runs."""

from __future__ import annotations

import torch


def test_legacy_module_imports() -> None:
    from pulse import legacy

    assert hasattr(legacy, "PulseConfig")
    assert hasattr(legacy, "PulseForCausalLM")


def test_legacy_smoke_forward() -> None:
    from pulse.legacy import PulseConfig, PulseForCausalLM

    cfg = PulseConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=32)
    model = PulseForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, labels=ids)
    assert torch.isfinite(out["loss"])
