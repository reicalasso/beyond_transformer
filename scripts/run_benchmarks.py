#!/usr/bin/env python3
"""
Benchmark Test Runner for PULSE Models
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from pulse.benchmarks.comprehensive_benchmark import ComprehensiveBenchmark


def create_test_model():
    """Create a simple test model for benchmarking."""
    class TestModel(nn.Module):
        def __init__(self, input_dim=128, output_dim=100):
            super().__init__()
            self.embedding = nn.Embedding(10000, 64)
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # Handle different input formats
            if x.dtype == torch.long:
                # Sequence input
                if x.dim() == 2:
                    embedded = self.embedding(x)  # [batch, seq, dim]
                    x = embedded.mean(dim=1)  # [batch, dim]
                else:
                    x = self.embedding(x).mean(dim=1)  # [batch, dim]
            else:
                # Feature input
                if x.dim() == 2:
                    if x.size(1) > 64:
                        x = x[:, :64]  # Truncate
                    elif x.size(1) < 64:
                        # Pad with zeros
                        padding = torch.zeros(x.size(0), 64 - x.size(1)).to(x.device)
                        x = torch.cat([x, padding], dim=1)
                else:
                    # Single feature, expand
                    x = x.unsqueeze(1).repeat(1, 64)
            
            x = self.relu(self.fc1(x))
            output = self.fc2(x)
            return output
    
    return TestModel()


def main():
    """Run benchmark tests."""
    print("PULSE - Benchmark Test Runner")
    print("=" * 50)
    
    # Create test model
    print("Creating test model...")
    model = create_test_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create benchmark suite
    print("Initializing benchmark suite...")
    benchmark_suite = ComprehensiveBenchmark(model, log_dir="benchmark_logs")
    
    # Run benchmarks
    print("Running benchmarks (quick test)...")
    results = benchmark_suite.run_all_benchmarks(batch_size=4, num_samples=50)
    
    # Print summary
    benchmark_suite.print_summary(results)
    
    print(f"\n🎉 Benchmark tests completed!")
    print(f"Logs saved to: {os.path.abspath('benchmark_logs')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
