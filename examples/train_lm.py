#!/usr/bin/env python3
"""
Example: Training an PULSE Language Model

This script demonstrates how to train an PULSE model for causal language modeling.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.utils.data import Dataset, DataLoader

from pulse.models.pulse_lm import PulseConfig, PulseForCausalLM
from pulse.training import PulseTrainer, TrainingArguments
from pulse.training.data_collator import DataCollatorForLanguageModeling


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""

    def __init__(self, texts: list, max_length: int = 128):
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Simple character-level tokenization
        input_ids = [ord(c) % 32000 for c in text[:self.max_length]]
        # Pad to max_length
        input_ids = input_ids + [0] * (self.max_length - len(input_ids))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long)}


def main():
    # Configuration
    config = PulseConfig(
        vocab_size=32000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_states=16,
        state_dim=256,
        intermediate_size=512,
        max_position_embeddings=256,
        dropout=0.1,
    )

    # Create model
    print("Creating model...")
    model = PulseForCausalLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create sample dataset
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "PULSEs are a new architecture for sequence modeling.",
        "This is an example of training a language model.",
        "Machine learning is transforming many industries.",
        "Deep learning models can learn complex patterns from data.",
    ] * 100  # Repeat for more training data

    train_dataset = SimpleTextDataset(sample_texts, max_length=64)
    eval_dataset = SimpleTextDataset(sample_texts[:10], max_length=64)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(pad_token_id=0)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output_lm_example",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-4,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        fp16=torch.cuda.is_available(),
    )

    # Create trainer
    trainer = PulseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    result = trainer.train()

    print(f"\nTraining completed!")
    print(f"Final step: {result['global_step']}")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    device = next(model.parameters()).device

    prompt = "The quick"
    input_ids = torch.tensor([[ord(c) % 32000 for c in prompt]], device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=50,
            temperature=0.8,
            top_k=50,
            do_sample=True,
        )

    output_text = "".join([chr(id % 128) for id in output_ids[0].tolist()])
    print(f"Generated: {output_text}")


if __name__ == "__main__":
    main()
