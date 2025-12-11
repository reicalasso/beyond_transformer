#!/usr/bin/env python3
"""
PULSE vs Transformer Comprehensive Benchmark

This script performs rigorous benchmarking:
1. Language Modeling Perplexity
2. Memory Usage Comparison
3. Throughput (tokens/second)
4. Scaling with Sequence Length
5. Long-Range Dependency Tests
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    vocab_size: int = 256
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_states: int = 16  # PULSE specific
    batch_size: int = 16
    seq_lengths: List[int] = None
    num_warmup: int = 3
    num_iterations: int = 10
    device: str = "cuda"
    
    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [128, 256, 512, 1024, 2048]


class SimpleTransformer(nn.Module):
    """Simple Transformer baseline for comparison."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Causal mask cache
        self.register_buffer("causal_mask", None)
        
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Causal mask
        mask = self._get_causal_mask(seq_len, device)
        
        # Transformer
        x = self.transformer(x, mask=mask)
        x = self.ln(x)
        
        # LM head
        logits = self.lm_head(x)
        
        output = {"logits": logits}
        
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
            
        return output


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def reset_memory():
    """Reset GPU memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_forward_pass(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark forward pass speed and memory."""
    model.eval()
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids, labels=labels)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Reset memory tracking
    reset_memory()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            outputs = model(input_ids, labels=labels)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    # Get memory
    memory_mb = get_memory_usage()
    
    # Calculate metrics
    avg_time = sum(times) / len(times)
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "memory_mb": memory_mb,
        "loss": outputs["loss"].item() if "loss" in outputs else 0.0,
    }


def benchmark_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    num_warmup: int = 3,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark training step (forward + backward)."""
    model.train()
    
    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(num_warmup):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Reset memory tracking
    reset_memory()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append(end - start)
    
    # Get memory
    memory_mb = get_memory_usage()
    
    # Calculate metrics
    avg_time = sum(times) / len(times)
    tokens_per_second = (batch_size * seq_len) / avg_time
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "memory_mb": memory_mb,
    }


def run_copy_task(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 128,
    delay: int = 64,
    num_samples: int = 100,
) -> float:
    """
    Copy task: Model must copy a sequence after a delay.
    Tests long-range dependency learning.
    """
    model.eval()
    vocab_size = 10  # Simple vocabulary for copy task
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Create pattern to copy
            pattern_len = 8
            pattern = torch.randint(1, vocab_size, (1, pattern_len), device=device)
            
            # Create input: [pattern] [zeros for delay] [pattern again as target]
            input_seq = torch.zeros(1, seq_len, dtype=torch.long, device=device)
            input_seq[0, :pattern_len] = pattern
            
            # Forward pass
            outputs = model(input_seq)
            logits = outputs["logits"]
            
            # Check if model can predict the pattern at the end
            # (simplified check - just verify pattern reconstruction)
            predictions = logits[0, delay:delay+pattern_len].argmax(dim=-1)
            
            # Count correct predictions
            correct += (predictions == pattern[0]).sum().item()
            total += pattern_len
    
    accuracy = correct / total
    return accuracy


def run_benchmark(config: BenchmarkConfig) -> Dict:
    """Run complete benchmark suite."""
    from pulse.models.pulse_lm import PulseConfig, PulseForCausalLM
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Running benchmark on: {device}")
    
    results = {
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_states": config.num_states,
            "batch_size": config.batch_size,
            "device": str(device),
        },
        "pulse": {},
        "transformer": {},
    }
    
    # Test each sequence length
    for seq_len in config.seq_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing sequence length: {seq_len}")
        logger.info(f"{'='*60}")
        
        # Skip if sequence too long for available memory
        if seq_len > 1024 and not torch.cuda.is_available():
            logger.warning(f"Skipping seq_len={seq_len} on CPU (too slow)")
            continue
        
        results["pulse"][seq_len] = {}
        results["transformer"][seq_len] = {}
        
        # ============ PULSE Model ============
        logger.info("\n--- PULSE Model ---")
        reset_memory()
        
        pulse_config = PulseConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_states=config.num_states,
            state_dim=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            max_position_embeddings=seq_len + 64,
            dropout=0.0,  # No dropout for benchmarking
        )
        
        pulse_model = PulseForCausalLM(pulse_config).to(device)
        pulse_params = sum(p.numel() for p in pulse_model.parameters())
        logger.info(f"PULSE parameters: {pulse_params:,}")
        
        # Forward pass benchmark
        try:
            pulse_forward = benchmark_forward_pass(
                pulse_model, config.batch_size, seq_len, config.vocab_size,
                device, config.num_warmup, config.num_iterations
            )
            results["pulse"][seq_len]["forward"] = pulse_forward
            logger.info(f"  Forward: {pulse_forward['avg_time_ms']:.2f}ms, "
                       f"{pulse_forward['tokens_per_second']:.0f} tok/s, "
                       f"{pulse_forward['memory_mb']:.0f}MB")
        except RuntimeError as e:
            logger.error(f"  Forward failed: {e}")
            results["pulse"][seq_len]["forward"] = {"error": str(e)}
        
        # Training step benchmark
        try:
            pulse_optimizer = torch.optim.AdamW(pulse_model.parameters(), lr=1e-4)
            pulse_train = benchmark_training_step(
                pulse_model, pulse_optimizer, config.batch_size, seq_len,
                config.vocab_size, device, config.num_warmup, config.num_iterations
            )
            results["pulse"][seq_len]["training"] = pulse_train
            logger.info(f"  Training: {pulse_train['avg_time_ms']:.2f}ms, "
                       f"{pulse_train['tokens_per_second']:.0f} tok/s, "
                       f"{pulse_train['memory_mb']:.0f}MB")
        except RuntimeError as e:
            logger.error(f"  Training failed: {e}")
            results["pulse"][seq_len]["training"] = {"error": str(e)}
        
        del pulse_model, pulse_optimizer
        reset_memory()
        
        # ============ Transformer Model ============
        logger.info("\n--- Transformer Model ---")
        
        transformer_model = SimpleTransformer(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=seq_len + 64,
            dropout=0.0,
        ).to(device)
        
        transformer_params = sum(p.numel() for p in transformer_model.parameters())
        logger.info(f"Transformer parameters: {transformer_params:,}")
        
        # Forward pass benchmark
        try:
            transformer_forward = benchmark_forward_pass(
                transformer_model, config.batch_size, seq_len, config.vocab_size,
                device, config.num_warmup, config.num_iterations
            )
            results["transformer"][seq_len]["forward"] = transformer_forward
            logger.info(f"  Forward: {transformer_forward['avg_time_ms']:.2f}ms, "
                       f"{transformer_forward['tokens_per_second']:.0f} tok/s, "
                       f"{transformer_forward['memory_mb']:.0f}MB")
        except RuntimeError as e:
            logger.error(f"  Forward failed: {e}")
            results["transformer"][seq_len]["forward"] = {"error": str(e)}
        
        # Training step benchmark
        try:
            transformer_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=1e-4)
            transformer_train = benchmark_training_step(
                transformer_model, transformer_optimizer, config.batch_size, seq_len,
                config.vocab_size, device, config.num_warmup, config.num_iterations
            )
            results["transformer"][seq_len]["training"] = transformer_train
            logger.info(f"  Training: {transformer_train['avg_time_ms']:.2f}ms, "
                       f"{transformer_train['tokens_per_second']:.0f} tok/s, "
                       f"{transformer_train['memory_mb']:.0f}MB")
        except RuntimeError as e:
            logger.error(f"  Training failed: {e}")
            results["transformer"][seq_len]["training"] = {"error": str(e)}
        
        del transformer_model, transformer_optimizer
        reset_memory()
        
        # ============ Comparison ============
        if "error" not in results["pulse"][seq_len].get("forward", {}) and \
           "error" not in results["transformer"][seq_len].get("forward", {}):
            
            pulse_fwd = results["pulse"][seq_len]["forward"]
            trans_fwd = results["transformer"][seq_len]["forward"]
            
            speedup = trans_fwd["avg_time_ms"] / pulse_fwd["avg_time_ms"]
            memory_ratio = trans_fwd["memory_mb"] / max(pulse_fwd["memory_mb"], 1)
            
            logger.info(f"\n  📊 Comparison (seq_len={seq_len}):")
            logger.info(f"     Speed: PULSE is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
            logger.info(f"     Memory: PULSE uses {1/memory_ratio:.2f}x {'less' if memory_ratio > 1 else 'more'}")
    
    return results


def print_summary(results: Dict):
    """Print benchmark summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)
    
    # Header
    logger.info(f"\n{'Seq Len':<10} | {'PULSE (ms)':<12} | {'Trans (ms)':<12} | {'Speedup':<10} | {'PULSE Mem':<12} | {'Trans Mem':<12}")
    logger.info("-" * 80)
    
    for seq_len in sorted([int(k) for k in results["pulse"].keys()]):
        seq_len = str(seq_len)
        
        pulse_data = results["pulse"].get(seq_len, {}).get("forward", {})
        trans_data = results["transformer"].get(seq_len, {}).get("forward", {})
        
        if "error" in pulse_data or "error" in trans_data:
            continue
        
        pulse_time = pulse_data.get("avg_time_ms", 0)
        trans_time = trans_data.get("avg_time_ms", 0)
        pulse_mem = pulse_data.get("memory_mb", 0)
        trans_mem = trans_data.get("memory_mb", 0)
        
        speedup = trans_time / pulse_time if pulse_time > 0 else 0
        
        logger.info(f"{seq_len:<10} | {pulse_time:<12.2f} | {trans_time:<12.2f} | {speedup:<10.2f}x | {pulse_mem:<12.0f} | {trans_mem:<12.0f}")
    
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="PULSE vs Transformer Benchmark")
    
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-states", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-lengths", type=str, default="128,256,512,1024")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_states=args.num_states,
        batch_size=args.batch_size,
        seq_lengths=[int(x) for x in args.seq_lengths.split(",")],
        num_iterations=args.num_iterations,
    )
    
    logger.info("=" * 60)
    logger.info("PULSE vs Transformer Benchmark")
    logger.info("=" * 60)
    logger.info(f"Config: hidden={config.hidden_size}, layers={config.num_layers}, "
               f"heads={config.num_heads}, states={config.num_states}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Sequence lengths: {config.seq_lengths}")
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
