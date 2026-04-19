#!/usr/bin/env python3
"""Train modern PULSE on TinyStories.

Usage::

    python scripts/train.py                                      # defaults
    python scripts/train.py --config configs/tinystories_small.yaml
    python scripts/train.py --hidden-size 512 --num-layers 12 --lr 5e-4

CLI flags always override YAML, which always overrides built-in defaults.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pulse import PulseConfig, PulseForCausalLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pulse.train")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (TinyStories, pre-tokenized, fixed-length windows)
# ─────────────────────────────────────────────────────────────────────────────


class TinyStoriesDataset(Dataset):
    """Pre-tokenized TinyStories chunked into fixed-length windows."""

    def __init__(self, data, pad_token_id: int, max_length: int = 256):
        self.pad = pad_token_id
        self.length = max_length
        self.examples: list[list[int]] = []

        stride = max_length // 2
        for item in tqdm(data, desc="Chunking"):
            tokens = item["input_ids"]
            for i in range(0, max(1, len(tokens) - max_length), stride):
                chunk = tokens[i : i + max_length + 1]
                if len(chunk) >= max_length // 2:
                    self.examples.append(chunk)
        log.info(f"Created {len(self.examples):,} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        tok = self.examples[idx]
        if len(tok) < self.length + 1:
            tok = tok + [self.pad] * (self.length + 1 - len(tok))
        else:
            tok = tok[: self.length + 1]
        t = torch.tensor(tok, dtype=torch.long)
        return t[:-1], t[1:]


def pretokenize(dataset, tokenizer, num_proc: int = 8):
    log.info(f"Pre-tokenizing with {num_proc} processes...")
    return dataset.map(
        lambda batch: tokenizer(batch["text"], add_special_tokens=True),
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schedules
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(
    step: int, warmup: int, total: int, max_lr: float, min_lr: float | None = None
) -> float:
    min_lr = min_lr if min_lr is not None else max_lr * 0.1
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_text(model, tokenizer, device, prompt: str, max_tokens: int = 80) -> str:
    was_training = model.training
    model.train(False)
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.train(was_training)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Train loop
# ─────────────────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU:  {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    log.info(f"Vocab: {len(tokenizer):,}  pad_id: {pad_id}")

    # Dataset
    log.info("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    log.info(f"Stories: {len(dataset):,}")

    dataset = pretokenize(dataset, tokenizer, num_proc=args.num_workers)
    split = dataset.train_test_split(test_size=0.02, seed=args.seed)

    train_ds = TinyStoriesDataset(split["train"], pad_id, args.seq_len)
    val_ds = TinyStoriesDataset(split["test"], pad_id, args.seq_len)
    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    cfg = PulseConfig(
        vocab_size=len(tokenizer),
        pad_token_id=pad_id,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_mult=args.ffn_mult,
        conv_kernel_size=args.conv_kernel_size,
        swa_every=args.swa_every,
        swa_window_size=args.swa_window_size,
        delta_chunk_size=args.delta_chunk_size,
        qk_norm=args.qk_norm,
        gate_bias_init=args.gate_bias_init,
        rope_max_seq_len=max(args.seq_len, args.rope_max_seq_len),
        max_seq_len=max(args.seq_len, args.rope_max_seq_len),
        dropout=args.dropout,
        logit_soft_cap=args.logit_soft_cap,
        z_loss_coef=args.z_loss_coef,
    )
    model = PulseForCausalLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Params: {n_params:,}  ({n_params * 4 / 1024**2:.1f} MB fp32)")
    log.info(f"Layers: {model.model.layer_types}")

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # Optimizer (decouple decay/no-decay groups)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ("bias", "norm", "embed")):
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )

    use_bf16 = args.bf16 or (
        not args.fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    use_fp16 = args.fp16 and not use_bf16
    autocast_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else None)
    scaler = GradScaler("cuda") if use_fp16 else None
    log.info(f"Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")

    # Resume
    step, best_val = 0, float("inf")
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scaler and ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            step = ckpt.get("step", 0)
            best_val = ckpt.get("val_loss", float("inf"))
            log.info(f"Resumed from step {step}, best_val={best_val:.4f}")
        else:
            log.warning(f"Checkpoint not found: {ckpt_path}")

    # Train loop
    log.info("=" * 70)
    log.info("Starting training...")
    log.info(f"  batch_size:    {args.batch_size}")
    log.info(f"  grad_accum:    {args.grad_accum}")
    log.info(f"  effective_bs:  {args.batch_size * args.grad_accum}")
    log.info(f"  max_steps:     {args.max_steps}")
    log.info(f"  warmup_steps:  {args.warmup_steps}")
    log.info(f"  lr:            {args.lr}")
    log.info(f"  seq_len:       {args.seq_len}")
    log.info("=" * 70)

    model.train()
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    accum_loss = 0.0
    tokens_seen = 0
    t0 = time.time()
    pbar = tqdm(total=args.max_steps, initial=step, desc="Training")

    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        tokens_seen += x.numel()

        mask = (x != pad_id).long()
        y = y.clone()
        y[y == pad_id] = -100

        lr = cosine_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        ctx = autocast("cuda", dtype=autocast_dtype) if autocast_dtype else contextlib.nullcontext()
        with ctx:
            out = model(x, labels=y, attention_mask=mask)
            loss = out["loss"] / args.grad_accum

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += out["loss"].item()

        if (step + 1) % args.grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()

        step += 1

        if step % 10 == 0:
            avg = accum_loss / 10
            accum_loss = 0.0
            tok_s = tokens_seen / max(1e-6, time.time() - t0)
            pbar.set_postfix(
                loss=f"{avg:.3f}",
                ppl=f"{math.exp(min(avg, 10)):.1f}",
                lr=f"{lr:.1e}",
                tok_s=f"{tok_s:.0f}",
            )
        pbar.update(1)

        if step % args.eval_steps == 0:
            model.train(False)
            val_losses: list[float] = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vmask = (vx != pad_id).long()
                    vy = vy.clone()
                    vy[vy == pad_id] = -100
                    with (
                        autocast("cuda", dtype=autocast_dtype)
                        if autocast_dtype
                        else contextlib.nullcontext()
                    ):
                        vout = model(vx, labels=vy, attention_mask=vmask)
                    val_losses.append(vout["loss"].item())
                    if len(val_losses) >= 100:
                        break

            val_loss = sum(val_losses) / len(val_losses)
            val_ppl = math.exp(min(val_loss, 10))
            log.info(f"\nStep {step}: val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

            sample = sample_text(model, tokenizer, device, "Once upon a time")
            log.info(f"Sample: {sample}")

            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "config": cfg.to_dict(),
                "val_loss": val_loss,
            }
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, output_dir / "best_model.pt")
                log.info(f"Saved best model (val_loss={val_loss:.4f})")
            torch.save(ckpt, output_dir / f"ckpt_{step:06d}.pt")
            model.train(True)

    pbar.close()
    elapsed = time.time() - t0
    log.info("=" * 70)
    log.info(
        f"Done! {elapsed / 3600:.2f}h  best_val={best_val:.4f}  "
        f"tok/s={tokens_seen / max(1e-6, elapsed):.0f}"
    )
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("PyYAML required for --config: pip install pyyaml") from e
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    flat: dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    return {k.replace("-", "_"): v for k, v in flat.items()}


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(description="Train modern PULSE on TinyStories")
    p.add_argument("--config", type=str, default=None, metavar="YAML")

    # Model
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=None)
    p.add_argument("--ffn-mult", type=float, default=2.7)
    p.add_argument("--conv-kernel-size", type=int, default=4)
    p.add_argument("--swa-every", type=int, default=4)
    p.add_argument("--swa-window-size", type=int, default=512)
    p.add_argument("--delta-chunk-size", type=int, default=64)
    p.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gate-bias-init", type=float, default=4.0)
    p.add_argument("--rope-max-seq-len", type=int, default=8192)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--logit-soft-cap", type=float, default=30.0)
    p.add_argument("--z-loss-coef", type=float, default=1.0e-4)

    # Training
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Precision
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--fp16", action="store_true", default=False)

    # Data
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=100_000)
    p.add_argument("--num-workers", type=int, default=4)

    # I/O
    p.add_argument("--output-dir", type=str, default="./output/pulse")
    p.add_argument("--resume", type=str, default=None)

    if pre_args.config:
        yaml_cfg = _load_yaml(pre_args.config)
        p.set_defaults(**{k: v for k, v in yaml_cfg.items() if v is not None})

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
