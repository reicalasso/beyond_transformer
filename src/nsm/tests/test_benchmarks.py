"""
Test script for all benchmark implementations
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

# Test the imports work
try:
    # Try absolute imports first
    from nsm.benchmarks.babi_benchmark import bAbIBenchmark, bAbIDataset
    from nsm.benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
    from nsm.benchmarks.lra_benchmark import LRABenchmark, LRADataset
    from nsm.benchmarks.pg19_benchmark import PG19Benchmark, PG19Dataset
except ImportError:
    # Fall back to relative imports
    from benchmarks.babi_benchmark import bAbIBenchmark, bAbIDataset
    from benchmarks.comprehensive_benchmark import ComprehensiveBenchmark
    from benchmarks.lra_benchmark import LRABenchmark, LRADataset
    from benchmarks.pg19_benchmark import PG19Benchmark, PG19Dataset


def test_lra_benchmark():
    """Test LRA benchmark."""
    print("Testing LRA Benchmark...")

    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, output_dim=10):
            super().__init__()
            self.embedding = nn.Embedding(1002, 64)
            self.fc = nn.Linear(64, output_dim)

        def forward(self, x):
            if x.dtype == torch.long:
                embedded = self.embedding(x)
            else:
                embedded = x.unsqueeze(-1).repeat(1, 1, 64)
            pooled = embedded.mean(dim=1)
            return self.fc(pooled)

    model = SimpleModel()
    benchmark = LRABenchmark(model)

    # Test one task
    results = benchmark.run_task("listops", batch_size=4, num_samples=50)
    print(f"  LRA ListOps results: {results}")

    print("✅ LRA Benchmark test passed\n")


def test_babi_benchmark():
    """Test bAbI benchmark."""
    print("Testing bAbI Benchmark...")

    # Create simple model
    class SimplebAbIModel(nn.Module):
        def __init__(self, output_dim=100):
            super().__init__()
            self.fc1 = nn.Linear(300, 128)
            self.fc2 = nn.Linear(128, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = SimplebAbIModel()
    benchmark = bAbIBenchmark(model)

    # Test one task
    results = benchmark.run_task(1, batch_size=4, num_samples=50)
    print(f"  bAbI Task 1 results: {results}")

    print("✅ bAbI Benchmark test passed\n")


def test_pg19_benchmark():
    """Test PG-19 benchmark."""
    print("Testing PG-19 Benchmark...")

    # Create simple model
    class SimplePG19Model(nn.Module):
        def __init__(self, output_dim=1000):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            if x.dim() == 2 and x.size(1) > 128:
                x = x[:, :128]  # Truncate
            elif x.dim() == 2 and x.size(1) < 128:
                # Pad
                padding = torch.zeros(x.size(0), 128 - x.size(1)).to(x.device)
                x = torch.cat([x, padding], dim=1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = SimplePG19Model()
    benchmark = PG19Benchmark(model)

    # Test language modeling
    lm_results = benchmark.run_language_modeling(
        batch_size=2, num_samples=20, seq_length=256
    )
    print(f"  PG-19 LM results: {lm_results}")

    print("✅ PG-19 Benchmark test passed\n")


def test_comprehensive_benchmark():
    """Test comprehensive benchmark suite."""
    print("Testing Comprehensive Benchmark Suite...")

    # Create simple model
    class TestModel(nn.Module):
        def __init__(self, output_dim=100):
            super().__init__()
            self.embedding = nn.Embedding(5000, 64)
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            # Handle different input types
            if x.dtype == torch.long:
                if x.dim() == 2:
                    embedded = self.embedding(x)
                    x = embedded.mean(dim=1)
                else:
                    x = self.embedding(x).mean(dim=1)
            elif x.dim() == 2 and x.size(1) > 64:
                x = x[:, :64]  # Truncate
            elif x.dim() == 2 and x.size(1) < 64:
                # Pad
                padding = torch.zeros(x.size(0), 64 - x.size(1)).to(x.device)
                x = torch.cat([x, padding], dim=1)

            x = self.relu(self.fc1(x))
            return self.fc2(x)

    model = TestModel()
    benchmark_suite = ComprehensiveBenchmark(model, log_dir="test_logs")

    # Run quick comprehensive test
    results = benchmark_suite.run_all_benchmarks(batch_size=4, num_samples=50)
    benchmark_suite.print_summary(results)

    print("✅ Comprehensive Benchmark Suite test passed\n")


def run_all_tests():
    """Run all benchmark tests."""
    print("Running All Benchmark Tests")
    print("=" * 50)

    try:
        test_lra_benchmark()
        test_babi_benchmark()
        test_pg19_benchmark()
        test_comprehensive_benchmark()

        print("=" * 50)
        print("🎉 All benchmark tests passed!")
        return True
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
