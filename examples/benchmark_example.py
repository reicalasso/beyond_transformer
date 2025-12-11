#!/usr/bin/env python3
"""
Example Benchmark Usage for PULSE Models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from pulse.benchmarks.comprehensive_benchmark import ComprehensiveBenchmark


class ExamplePulseModel(nn.Module):
    """
    Example PULSE Model for benchmarking.
    """
    
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Forward pass supporting different input formats.
        
        Args:
            x: Input tensor (can be indices or features)
            
        Returns:
            Output tensor
        """
        # Handle sequence inputs (long sequences)
        if x.dtype == torch.long:
            if x.dim() == 2:  # [batch_size, seq_len]
                embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
                x = embedded.mean(dim=1)  # [batch_size, embed_dim]
            else:  # [batch_size]
                x = self.embedding(x)  # [batch_size, embed_dim]
        else:
            # Handle feature inputs
            if x.dim() == 2:
                if x.size(1) > 128:
                    x = x[:, :128]  # Truncate to embed_dim
                elif x.size(1) < 128:
                    # Pad with zeros
                    padding = torch.zeros(x.size(0), 128 - x.size(1)).to(x.device)
                    x = torch.cat([x, padding], dim=1)
            else:
                # Single feature, expand
                x = x.unsqueeze(1).repeat(1, 128)
        
        # Process through layers
        x = self.dropout(self.relu(self.fc1(x)))
        output = self.fc2(x)
        return output


def run_example_benchmarks():
    """
    Run example benchmarks to demonstrate usage.
    """
    print("PULSE - Benchmark Example")
    print("=" * 50)
    
    # Create example model
    print("Creating example PULSE model...")
    model = ExamplePulseModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create benchmark suite
    print("Initializing benchmark suite...")
    benchmark_suite = ComprehensiveBenchmark(model, log_dir="example_benchmark_logs")
    
    # Run quick demonstration benchmarks
    print("Running demonstration benchmarks...")
    results = benchmark_suite.run_all_benchmarks(batch_size=4, num_samples=50)
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*50)
    benchmark_suite.print_summary(results)
    
    print(f"\n\x1b[92m\x1b[1m\xf0\x9f\xa5\xb3 Example benchmarks completed!\x1b[0m")
    print(f"Logs saved to: example_benchmark_logs/")
    
    return results


if __name__ == "__main__":
    # Run example
    results = run_example_benchmarks()
    
    # Show sample results structure
    print("\n" + "="*50)
    print("SAMPLE RESULTS STRUCTURE")
    print("="*50)
    
    if 'lra_results' in results and results['lra_results']:
        print("LRA Results Sample:")
        for task, task_results in list(results['lra_results'].items())[:2]:
            print(f"  {task}: {task_results}")
    
    if 'babi_results' in results and results['babi_results']:
        print("\nbAbI Results Sample:")
        for task_id, task_results in list(results['babi_results'].items())[:2]:
            print(f"  Task {task_id}: {task_results}")
