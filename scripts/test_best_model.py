#!/usr/bin/env python3
"""Test a trained modern PULSE checkpoint with generation and perplexity probes.

Usage:
    python scripts/test_best_model.py
    python scripts/test_best_model.py --checkpoint output/pulse/best_model.pt
    python scripts/test_best_model.py --checkpoint output/pulse/best_model.pt --max-tokens 200
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoTokenizer

from pulse import PulseConfig, PulseForCausalLM

PROMPTS = [
    "Once upon a time",
    "The little girl",
    "Tom was a boy who",
    "In the forest there was",
    "The cat and the dog",
    "One day, a little bird",
    "Mom said to her son",
    "The big bear went to",
]

PPL_SENTENCES = [
    "Once upon a time there was a little girl.",
    "He was very happy to see his friend.",
    "The dog ran fast in the park.",
    "She liked to play with her toys every day.",
    "The cat sat on the mat.",
]


def load_model(checkpoint_path: str, device: str):
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(ckpt_dir, "config.json")

    with open(config_path) as f:
        config_dict = json.load(f)
    config = PulseConfig.from_dict(config_dict)

    model = PulseForCausalLM(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.train(False)

    n = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded: {n:,} params  |  val_loss={ckpt.get('val_loss', '?'):.4f}  step={ckpt.get('step', '?')}"
    )
    print(f"  hidden={config.hidden_size}  layers={config.num_layers}  heads={config.num_heads}")
    print(f"  layer_types={model.model.layer_types}")
    print(f"  swa_window={config.swa_window_size}  delta_chunk={config.delta_chunk_size}\n")
    return model, config


def run_generation(
    model, tokenizer, device, prompts, max_tokens, temperature, top_k, top_p, rep_penalty
):
    print("=" * 70)
    print(f"temp={temperature}  top_k={top_k}  top_p={top_p}  rep_penalty={rep_penalty}")
    print("=" * 70)
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.no_grad(), ctx:
            out = model.generate(
                ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        print(f'\n["{prompt}"]\n{tokenizer.decode(out[0], skip_special_tokens=True)}')
    print()


def run_perplexity(model, tokenizer, device, sentences):
    print("Perplexity:")
    use_amp = device == "cuda" and torch.cuda.is_bf16_supported()
    for sent in sentences:
        ids = tokenizer.encode(sent, return_tensors="pt").to(device)
        if ids.shape[1] < 2:
            continue
        x, y = ids[:, :-1], ids[:, 1:]
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if use_amp
            else torch.cuda.amp.autocast(enabled=False)
        )
        with torch.no_grad(), ctx:
            out = model(x, labels=y)
        ppl = torch.exp(out["loss"]).item()
        print(f"  PPL={ppl:7.2f}  {sent[:70]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="output/pulse/best_model.pt")
    p.add_argument("--max-tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rep-penalty", type=float, default=1.15)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model, config = load_model(args.checkpoint, device)

    run_generation(
        model,
        tokenizer,
        device,
        PROMPTS,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.rep_penalty,
    )

    for temp in [0.5, 1.0]:
        run_generation(
            model,
            tokenizer,
            device,
            PROMPTS[:3],
            80,
            temp,
            args.top_k,
            args.top_p,
            args.rep_penalty,
        )

    run_perplexity(model, tokenizer, device, PPL_SENTENCES)
    print("\nDone.")


if __name__ == "__main__":
    main()
