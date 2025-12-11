#!/usr/bin/env python3
"""
Example Performance Measurement Usage
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from pulse.utils.comparative_analyzer import ComparativePerformanceAnalyzer


class ExampleModelA(nn.Module):
    """Simple example model A."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ExampleModelB(nn.Module):
    """Simple example model B."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def run_performance_example():
    """Run performance measurement example."""
    print("PULSE - Performance Measurement Example")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ComparativePerformanceAnalyzer("example_performance_logs")
    
    # Define models to compare
    models = {
        "SimpleModelA": (ExampleModelA(), (32, 128)),
        "SimpleModelB": (ExampleModelB(), (32, 128))
    }
    
    print("Comparing models...")
    results = analyzer.compare_models(models, num_iterations=3)
    
    # Print results
    analyzer.print_comparison_summary()
    
    # Save results
    results_file = analyzer.save_comparison_results()
    print(f"\nResults saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_performance_example()
    print("\n🎉 Performance measurement example completed!")
