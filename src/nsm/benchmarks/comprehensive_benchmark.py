"""
Comprehensive Benchmark Suite for Neural State Machine Models

This module implements a comprehensive benchmark suite that runs LRA, bAbI, and PG-19 benchmarks.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .babi_benchmark import bAbIBenchmark, bAbIDataset
from .lra_benchmark import LRABenchmark, LRADataset
from .pg19_benchmark import PG19Benchmark, PG19Dataset


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite for Neural State Machine Models.
    """

    def __init__(self, model, log_dir: str = "benchmark_logs"):
        """
        Initialize comprehensive benchmark suite.

        Args:
            model: NSM model to benchmark
            log_dir: Directory to save benchmark logs
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize individual benchmarks
        self.lra_benchmark = LRABenchmark(model, self.device)
        self.babi_benchmark = bAbIBenchmark(model, self.device)
        self.pg19_benchmark = PG19Benchmark(model, self.device)

    def run_lra_benchmarks(
        self, batch_size: int = 32, num_samples: int = 1000
    ) -> Dict[str, Dict[str, float]]:
        """
        Run all LRA benchmarks.

        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task

        Returns:
            Dictionary with LRA benchmark results
        """
        print("\n" + "=" * 60)
        print("RUNNING LRA BENCHMARKS")
        print("=" * 60)

        return self.lra_benchmark.run_all_tasks(batch_size, num_samples)

    def run_babi_benchmarks(
        self, batch_size: int = 32, num_samples: int = 1000
    ) -> Dict[int, Dict[str, float]]:
        """
        Run key bAbI benchmarks.

        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task

        Returns:
            Dictionary with bAbI benchmark results
        """
        print("\n" + "=" * 60)
        print("RUNNING bAbI BENCHMARKS")
        print("=" * 60)

        return self.babi_benchmark.run_key_tasks(batch_size, num_samples)

    def run_pg19_benchmarks(
        self, batch_size: int = 8, num_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Run PG-19 benchmarks.

        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task

        Returns:
            Dictionary with PG-19 benchmark results
        """
        print("\n" + "=" * 60)
        print("RUNNING PG-19 BENCHMARKS")
        print("=" * 60)

        results = {}

        # Run language modeling benchmark
        lm_results = self.pg19_benchmark.run_language_modeling(
            batch_size=batch_size, num_samples=num_samples, seq_length=2048
        )
        results["language_modeling"] = lm_results

        # Run long-context memory benchmark
        memory_results = self.pg19_benchmark.run_long_context_memory(
            batch_size=batch_size // 2,
            num_samples=num_samples // 2,
            context_length=4096,
        )
        results["long_context_memory"] = memory_results

        return results

    def run_all_benchmarks(
        self, batch_size: int = 32, num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task

        Returns:
            Dictionary with all benchmark results
        """
        print("Running Comprehensive Benchmark Suite")
        print("=" * 60)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Using device: {self.device}")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "device": str(self.device),
            },
            "lra_results": {},
            "babi_results": {},
            "pg19_results": {},
        }

        # Run LRA benchmarks
        try:
            results["lra_results"] = self.run_lra_benchmarks(
                batch_size=min(batch_size, 16),  # Smaller batch for LRA
                num_samples=num_samples // 10,  # Fewer samples for speed
            )
        except Exception as e:
            print(f"Error in LRA benchmarks: {e}")
            results["lra_results"] = {"error": str(e)}

        # Run bAbI benchmarks
        try:
            results["babi_results"] = self.run_babi_benchmarks(
                batch_size=batch_size,
                num_samples=num_samples // 5,  # Fewer samples for speed
            )
        except Exception as e:
            print(f"Error in bAbI benchmarks: {e}")
            results["babi_results"] = {"error": str(e)}

        # Run PG-19 benchmarks
        try:
            results["pg19_results"] = self.run_pg19_benchmarks(
                batch_size=min(batch_size, 8),  # Smaller batch for PG-19
                num_samples=num_samples // 10,  # Fewer samples for speed
            )
        except Exception as e:
            print(f"Error in PG-19 benchmarks: {e}")
            results["pg19_results"] = {"error": str(e)}

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: Dict[str, Any]):
        """
        Save benchmark results to file.

        Args:
            results: Benchmark results dictionary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"benchmark_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print benchmark summary.

        Args:
            results: Benchmark results dictionary
        """
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Model Parameters: {results['model_info']['parameters']:,}")
        print(f"Device: {results['model_info']['device']}")

        # LRA Results
        print("\nLRA RESULTS:")
        if "error" in results["lra_results"]:
            print(f"  ❌ Error: {results['lra_results']['error']}")
        else:
            for task, task_results in results["lra_results"].items():
                if "error" in task_results:
                    print(f"  {task}: ❌ Error: {task_results['error']}")
                else:
                    accuracy = task_results.get("accuracy", 0.0)
                    loss = task_results.get("loss", 0.0)
                    print(f"  {task}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")

        # bAbI Results
        print("\nbAbI RESULTS:")
        if "error" in results["babi_results"]:
            print(f"  ❌ Error: {results['babi_results']['error']}")
        else:
            for task_id, task_results in results["babi_results"].items():
                if "error" in task_results:
                    print(f"  Task {task_id}: ❌ Error: {task_results['error']}")
                else:
                    accuracy = task_results.get("accuracy", 0.0)
                    loss = task_results.get("loss", 0.0)
                    print(f"  Task {task_id}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")

        # PG-19 Results
        print("\nPG-19 RESULTS:")
        if "error" in results["pg19_results"]:
            print(f"  ❌ Error: {results['pg19_results']['error']}")
        else:
            for task_name, task_results in results["pg19_results"].items():
                if "error" in task_results:
                    print(f"  {task_name}: ❌ Error: {task_results['error']}")
                else:
                    if task_name == "language_modeling":
                        perplexity = task_results.get("perplexity", 0.0)
                        accuracy = task_results.get("accuracy", 0.0)
                        print(
                            f"  {task_name}: Perplexity={perplexity:.4f}, Accuracy={accuracy:.4f}"
                        )
                    elif task_name == "long_context_memory":
                        memory_score = task_results.get("memory_score", 0.0)
                        print(f"  {task_name}: Memory Score={memory_score:.4f}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Comprehensive Benchmark Suite...")

    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self, input_dim=1000, output_dim=100):
            super().__init__()
            self.embedding = nn.Embedding(10000, 128)
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            # Handle both long sequences and regular inputs
            if x.dim() == 2 and x.size(1) > 100:  # Long sequence
                # Process long sequence
                if x.dtype == torch.long:
                    embedded = self.embedding(x)  # [batch_size, seq_len, 128]
                else:
                    embedded = x.unsqueeze(-1).repeat(
                        1, 1, 128
                    )  # [batch_size, seq_len, 128]
                pooled = embedded.mean(dim=1)  # [batch_size, 128]
                x = pooled
            elif x.dim() == 2:  # Regular batch of features
                if x.size(1) > 128:  # Need to reduce dimension
                    x = x[:, :128]  # Take first 128 features
                # Pad if necessary
                if x.size(1) < 128:
                    padding = torch.zeros(x.size(0), 128 - x.size(1)).to(x.device)
                    x = torch.cat([x, padding], dim=1)

            x = self.relu(self.fc1(x))
            output = self.fc2(x)
            return output

    # Test with simple model
    model = TestModel()
    benchmark_suite = ComprehensiveBenchmark(model, log_dir="test_benchmark_logs")

    # Run quick benchmark
    print("Running quick benchmark test...")
    results = benchmark_suite.run_all_benchmarks(batch_size=8, num_samples=100)

    # Print summary
    benchmark_suite.print_summary(results)

    print("\n✅ Comprehensive Benchmark Suite test completed!")
