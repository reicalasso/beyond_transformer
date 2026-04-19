#!/usr/bin/env python3
"""Microbenchmark harness for the modern PULSE architecture.

Reports forward latency, training-step latency (forward + backward + step),
peak-memory (CUDA only), and per-token incremental decode latency for a
given configuration. CPU and CUDA both supported; defaults are tiny so it
runs in a few seconds on any machine.

Examples:
    python scripts/bench.py
    python scripts/bench.py --hidden-size 768 --num-layers 12 --seq-len 1024
    python scripts/bench.py --device cpu --reps 3
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from pulse import PulseConfig, PulseForCausalLM


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_repeat(fn, reps: int, device: torch.device) -> dict[str, float]:
    times: list[float] = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--swa-every", type=int, default=2)
    p.add_argument("--swa-window-size", type=int, default=64)
    p.add_argument("--delta-chunk-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--decode-tokens", type=int, default=32)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    cfg = PulseConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        swa_every=args.swa_every,
        swa_window_size=args.swa_window_size,
        delta_chunk_size=args.delta_chunk_size,
        max_seq_len=max(args.seq_len, args.decode_tokens) * 2,
        rope_max_seq_len=max(args.seq_len, args.decode_tokens) * 2,
    )
    model = PulseForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Params: {n_params:,}  ({n_params * 4 / 1024**2:.2f} MB fp32)  "
        f"layers={model.model.layer_types}"
    )

    ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device)

    # ── Warmup ────────────────────────────────────────────────────────
    for _ in range(args.warmup):
        with torch.no_grad():
            _ = model(ids)

    # ── Forward (no grad) ────────────────────────────────────────────
    def forward():
        with torch.no_grad():
            _ = model(ids)

    fwd = _time_repeat(forward, args.reps, device)
    tok_per_s_fwd = (args.batch_size * args.seq_len) / fwd["median"]

    # ── Train step ───────────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=1.0e-4)

    def step():
        optim.zero_grad()
        out = model(ids, labels=ids)
        out["loss"].backward()
        optim.step()

    for _ in range(args.warmup):
        step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train = _time_repeat(step, args.reps, device)
    tok_per_s_train = (args.batch_size * args.seq_len) / train["median"]
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0

    # ── Incremental decode ───────────────────────────────────────────
    prompt = ids[:, : args.seq_len // 2]

    def decode():
        with torch.no_grad():
            out = model(prompt)
            states = out["layer_states"]
            tok = out["logits"][:, -1].argmax(-1, keepdim=True)
            for _ in range(args.decode_tokens - 1):
                out = model(tok, layer_states=states)
                states = out["layer_states"]
                tok = out["logits"][:, -1].argmax(-1, keepdim=True)

    for _ in range(args.warmup):
        decode()
    dec = _time_repeat(decode, args.reps, device)
    decode_tok_s = (args.batch_size * args.decode_tokens) / dec["median"]

    # ── Report ────────────────────────────────────────────────────────
    def fmt(d: dict[str, float]) -> str:
        return f"med={d['median'] * 1000:.2f}ms  mean={d['mean'] * 1000:.2f}ms  ±{d['stdev'] * 1000:.2f}ms"

    print()
    print(f"Forward            B={args.batch_size} T={args.seq_len}")
    print(f"  {fmt(fwd)}   {tok_per_s_fwd:,.0f} tok/s")
    print("Train step (fwd+bwd+opt)")
    print(f"  {fmt(train)}   {tok_per_s_train:,.0f} tok/s")
    if device.type == "cuda":
        print(f"  peak memory: {peak_mb:,.1f} MB")
    print(f"Decode (prompt={prompt.shape[1]}, +{args.decode_tokens})")
    print(f"  {fmt(dec)}   {decode_tok_s:,.0f} tok/s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
