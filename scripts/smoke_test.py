#!/usr/bin/env python3
"""
Minimal sanity check for PULSE v3.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from pulse import PulseConfig, PulseForCausalLM


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = PulseConfig(
        vocab_size=32000,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        fast_decay=0.70,
        slow_decay=0.97,
        max_seq_len=128,
    )

    model = PulseForCausalLM(config).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")

    batch_size, seq_len = 2, 32
    dummy = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        out = model(dummy, labels=dummy)

    print(f"logits : {out['logits'].shape}")
    print(f"loss   : {out['loss'].item():.4f}")

    generated = model.generate(dummy[:, :4], max_new_tokens=8)
    print(f"generated shape: {generated.shape}")

    print("OK — smoke test passed.")


if __name__ == "__main__":
    main()
