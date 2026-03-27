#!/usr/bin/env python3
"""
Test the best PULSE model checkpoint with various generation prompts.

Usage:
    python scripts/test_best_model.py
    python scripts/test_best_model.py --checkpoint output/pulse_tinystories/best_model.pt
    python scripts/test_best_model.py --checkpoint output/pulse_4090/best_model.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pulse.models import PulseConfig, PulseForCausalLM


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


def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"

    with open(config_path) as f:
        config_dict = json.load(f)
    config = PulseConfig(**config_dict)

    model = PulseForCausalLM(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} params")
    print(f"  hidden={config.hidden_size}, layers={config.num_layers}, heads={config.num_heads}")
    print(f"  val_loss={checkpoint.get('val_loss', '?')}, step={checkpoint.get('step', '?')}")
    return model, config


def test_generation(model, tokenizer, device, prompts, max_tokens=150,
                    temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2):
    """Generate from multiple prompts and print results."""
    print("\n" + "=" * 70)
    print(f"Generation settings: temp={temperature}, top_k={top_k}, "
          f"top_p={top_p}, rep_penalty={repetition_penalty}")
    print("=" * 70)

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n--- Prompt: \"{prompt}\" ---")
        print(text)

    print("\n" + "=" * 70)


def test_perplexity(model, tokenizer, device, sentences):
    """Compute per-sentence perplexity."""
    print("\nPerplexity on test sentences:")
    for sent in sentences:
        tokens = tokenizer.encode(sent, return_tensors="pt").to(device)
        if tokens.shape[1] < 2:
            continue
        x, y = tokens[:, :-1], tokens[:, 1:]
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(x, labels=y)
        ppl = torch.exp(torch.tensor(outputs["loss"])).item()
        print(f"  PPL={ppl:7.2f}  | {sent[:80]}")


def main():
    parser = argparse.ArgumentParser(description="Test best PULSE model")
    parser.add_argument("--checkpoint", type=str,
                        default="output/pulse_tinystories/best_model.pt")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-penalty", type=float, default=1.2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Test generation with default settings
    test_generation(model, tokenizer, device, PROMPTS,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.rep_penalty)

    # Test with different temperatures
    for temp in [0.5, 1.0]:
        print(f"\n{'─' * 70}")
        print(f"Temperature sweep: {temp}")
        test_generation(model, tokenizer, device, PROMPTS[:3],
                        max_tokens=100, temperature=temp,
                        repetition_penalty=args.rep_penalty)

    # Perplexity on simple sentences
    ppl_sentences = [
        "The cat sat on the mat.",
        "Once upon a time there was a little girl.",
        "He was very happy to see his friend.",
        "The dog ran fast in the park.",
        "She liked to play with her toys every day.",
    ]
    test_perplexity(model, tokenizer, device, ppl_sentences)

    print("\nDone!")


if __name__ == "__main__":
    main()
