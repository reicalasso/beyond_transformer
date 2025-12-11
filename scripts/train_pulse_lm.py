#!/usr/bin/env python3
"""
PULSE Language Model Training Script

This script trains a PULSE model on text data for language modeling.
Supports WikiText-2, custom text files, or synthetic data for quick testing.
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pulse.models.pulse_lm import PulseConfig, PulseForCausalLM

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(
        self,
        text: str,
        block_size: int = 128,
        vocab_size: int = 256,
    ):
        """
        Initialize dataset.

        Args:
            text: Raw text data
            block_size: Sequence length for training
            vocab_size: Vocabulary size (256 for byte-level)
        """
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Byte-level tokenization (simple but effective)
        self.data = torch.tensor(
            [min(ord(c), vocab_size - 1) for c in text],
            dtype=torch.long
        )

        # Calculate number of samples
        self.num_samples = max(1, (len(self.data) - block_size) // block_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1

        chunk = self.data[start:end]

        # Pad if necessary
        if len(chunk) < self.block_size + 1:
            chunk = torch.cat([
                chunk,
                torch.zeros(self.block_size + 1 - len(chunk), dtype=torch.long)
            ])

        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def load_wikitext2_hf():
    """Load WikiText-2 using Hugging Face datasets."""
    try:
        from datasets import load_dataset
        logger.info("Loading WikiText-2 from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])
        return train_text, val_text
    except ImportError:
        logger.warning("datasets not installed. Using synthetic data.")
        return None, None
    except Exception as e:
        logger.warning(f"Could not load WikiText-2: {e}. Using synthetic data.")
        return None, None


def load_text_data(data_path: str = None, use_wikitext: bool = True) -> tuple:
    """Load text data for training."""
    if use_wikitext:
        train_text, val_text = load_wikitext2_hf()
        if train_text is not None:
            return train_text, val_text
        # Fallback to synthetic
        logger.info("Falling back to synthetic data...")
        use_wikitext = False
    
    if data_path:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Split 90/10
        split_idx = int(len(text) * 0.9)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
    else:
        # Generate richer synthetic data
        logger.info("Using synthetic data for training...")
        paragraphs = [
            "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet. It has been used for typing practice for many years. The fox is known for its speed and agility. ",
            "Machine learning is transforming the world of technology. Deep neural networks can learn complex patterns from data. These models are used in image recognition, natural language processing, and many other applications. ",
            "Artificial intelligence is advancing rapidly in recent years. Researchers are developing new algorithms that can solve increasingly complex problems. The future of AI holds great promise for humanity. ",
            "Natural language processing enables computers to understand human language. This technology powers virtual assistants, translation services, and text analysis tools. NLP has become essential in modern applications. ",
            "Computer vision allows machines to interpret visual information from the world. This includes recognizing objects, faces, and scenes in images and videos. Self-driving cars rely heavily on computer vision. ",
            "Reinforcement learning teaches agents through trial and error. The agent receives rewards for good actions and penalties for bad ones. This approach has achieved superhuman performance in many games. ",
            "The history of computing began with mechanical calculators. Charles Babbage designed the first programmable computer in the 1800s. Today, computers are billions of times more powerful than early machines. ",
            "Data science combines statistics, programming, and domain expertise. Data scientists analyze large datasets to extract insights and make predictions. This field has become crucial for business decision making. ",
            "The internet has revolutionized how we communicate and share information. Billions of people around the world are now connected. Social media platforms have changed the way we interact with each other. ",
            "Quantum computing promises to solve problems that are impossible for classical computers. These machines use quantum bits that can exist in multiple states simultaneously. Major tech companies are racing to build practical quantum computers. ",
        ]
        train_text = " ".join(paragraphs * 500)
        val_text = " ".join(paragraphs * 50)

    return train_text, val_text


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"] / gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs["loss"].item()
        num_batches += 1

        # Update progress bar
        avg_loss = total_loss / num_batches
        perplexity = math.exp(min(avg_loss, 20))
        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "ppl": f"{perplexity:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20))

    return avg_loss, perplexity


@torch.no_grad()
def generate_sample(
    model: nn.Module,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    device: torch.device = None,
) -> str:
    """Generate text sample from prompt."""
    model.eval()

    # Tokenize prompt
    input_ids = torch.tensor(
        [[min(ord(c), 255) for c in prompt]],
        dtype=torch.long,
        device=device,
    )

    # Generate
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.9,
        do_sample=True,
    )

    # Decode
    output_text = "".join([chr(min(id, 127)) for id in output_ids[0].tolist()])
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Train PULSE Language Model")

    # Data arguments
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--use-wikitext", action="store_true", default=True,
                        help="Use WikiText-2 dataset")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for quick testing")

    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-states", type=int, default=16, help="Number of states")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--gradient-accumulation", type=int, default=1,
                        help="Gradient accumulation steps")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./output/pulse_lm",
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    if args.synthetic:
        train_text, val_text = load_text_data(use_wikitext=False)
    elif args.data:
        train_text, val_text = load_text_data(data_path=args.data, use_wikitext=False)
    else:
        train_text, val_text = load_text_data(use_wikitext=True)

    logger.info(f"Train size: {len(train_text):,} chars")
    logger.info(f"Val size: {len(val_text):,} chars")

    # Create datasets
    train_dataset = TextDataset(train_text, block_size=args.block_size, vocab_size=args.vocab_size)
    val_dataset = TextDataset(val_text, block_size=args.block_size, vocab_size=args.vocab_size)

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Val samples: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    logger.info("Creating model...")
    config = PulseConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_states=args.num_states,
        state_dim=args.hidden_size,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.block_size + 64,
        dropout=0.1,
        use_adaptive_states=True,
    )

    model = PulseForCausalLM(config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.1,
    )

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            gradient_accumulation_steps=args.gradient_accumulation,
        )

        # Evaluate
        val_loss, val_ppl = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

        # Generate sample
        sample = generate_sample(model, "The ", max_length=50, device=device)
        logger.info(f"Sample: {sample[:100]}...")

        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config.to_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, output_dir / "best_model.pt")
                logger.info(f"Saved best model (val_loss={val_loss:.4f})")

            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # Training complete
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Training complete in {elapsed/60:.1f} minutes")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    logger.info("=" * 60)

    # Final generation samples
    logger.info("\nGeneration samples:")
    prompts = ["The ", "Machine learning ", "In the ", "Once upon a "]
    for prompt in prompts:
        sample = generate_sample(model, prompt, max_length=80, device=device)
        logger.info(f"  '{prompt}' -> {sample}")


if __name__ == "__main__":
    main()
