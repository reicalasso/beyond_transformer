#!/usr/bin/env python3
"""Minimal end-to-end sanity check for modern PULSE.

Builds a tiny config, runs forward + loss + generate, and prints shapes/numbers.
Exits non-zero if anything is non-finite.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from pulse import PulseConfig, PulseForCausalLM


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = PulseConfig(
        vocab_size=512,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        ffn_mult=2.0,
        conv_kernel_size=3,
        swa_every=2,
        swa_window_size=32,
        delta_chunk_size=16,
        max_seq_len=256,
        rope_max_seq_len=256,
    )

    model = PulseForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params:  {n_params:,}  ({n_params * 4 / 1024**2:.2f} MB fp32)")
    print(f"Layers:  {model.model.layer_types}")

    ids = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
    out = model(ids, labels=ids)
    print(f"logits:  {tuple(out['logits'].shape)}")
    print(
        f"loss:    {out['loss'].item():.4f}  (ce={out['ce_loss'].item():.4f}, "
        f"z={out['z_loss'].item():.6f})"
    )
    assert torch.isfinite(out["loss"]), "non-finite loss"

    gen = model.generate(ids[:, :4], max_new_tokens=8, top_k=10, top_p=0.9)
    print(f"gen:     {tuple(gen.shape)}")
    assert gen.shape == (2, 12)

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
