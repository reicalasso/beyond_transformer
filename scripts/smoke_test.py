#!/usr/bin/env python3
"""
Minimal sanity check for the PULSE model on GPU.

Usage (from repo root, after activating venv):

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

    if device.type != "cuda":
        print("WARNING: CUDA device not detected. Test will run on CPU.")

    # Small config so the test is lightweight
    config = PulseConfig(
        vocab_size=32000,
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        max_seq_len=128,
    )

    model = PulseForCausalLM(config).to(device)
    model.eval()

    # Dummy batch: [batch, seq_len]
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        out = model(dummy_input, labels=dummy_input)

    logits = out["logits"]
    loss = out.get("loss")

    print(f"logits.shape = {logits.shape}")
    if loss is not None:
        print(f"loss = {loss.item():.4f}")

    # Quick generation check
    generated = model.generate(dummy_input[:, :4], max_new_tokens=8)
    print(f"generated.shape = {generated.shape}")

    print("OK — smoke test passed.")


if __name__ == "__main__":
    main()

