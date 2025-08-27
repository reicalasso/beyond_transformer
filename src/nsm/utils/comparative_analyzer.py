"""
Comparative Performance Analysis for Neural State Machine Models

This module provides tools for comparative performance analysis and table updates.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json
from datetime import datetime

from nsm.utils.performance_monitor import PerformanceMonitor, ModelProfiler


class ComparativePerformanceAnalyzer:
    """
    Analyze and compare performance of different models.
    """
    
    def __init__(self, log_dir: str = "comparative_analysis"):
        """
        Initialize comparative analyzer.
        
        Args:
            log_dir: Directory to save analysis results
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.performance_monitor = PerformanceMonitor(log_dir)
        self.model_profiler = ModelProfiler()
        self.comparison_results = []
        self.debugger = None
    
    def set_debugger(self, debugger):
        """
        Set debugger for this analyzer.
        
        Args:
            debugger: NSMDebugger instance
        """
        self.debugger = debugger
    
    def benchmark_model(self, model: nn.Module, model_name: str, 
                       input_shape: Tuple[int, ...], 
                       num_iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark a single model.
        
        Args:
            model: Model to benchmark
            model_name: Name of the model
            input_shape: Input tensor shape
            num_iterations: Number of iterations for averaging
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking {model_name}...")
        
        # Log to debugger if available
        if self.debugger is not None:
            self.debugger.log_step("benchmark_start", {
                'model_name': model_name,
                'input_shape': input_shape,
                'num_iterations': num_iterations
            })
        
        # Profile model
        profile = self.model_profiler.profile_model(model, input_shape)
        
        # Create sample input
        sample_input = torch.randn(input_shape)
        # Create appropriate target shape
        if len(input_shape) > 1:
            target_shape = (input_shape[0], max(1, profile['total_parameters'] // 10000))
        else:
            target_shape = (input_shape[0], 1)
        sample_target = torch.randn(target_shape)
        
        # Initialize performance monitor
        self.performance_monitor.start_monitoring()
        
        # Measure multiple forward passes
        forward_times = []
        for i in range(num_iterations):
            metrics = self.performance_monitor.measure_forward_pass(model, sample_input)
            forward_times.append(metrics['forward_time'])
            
            # Log to debugger if available
            if self.debugger is not None:
                self.debugger.log_step(f"forward_pass_{i}", metrics)
        
        # Measure multiple backward passes
        backward_times = []
        for i in range(num_iterations):
            metrics = self.performance_monitor.measure_backward_pass(model, sample_input, sample_target)
            backward_times.append(metrics['backward_time'])
            
            # Log to debugger if available
            if self.debugger is not None:
                self.debugger.log_step(f"backward_pass_{i}", metrics)
        
        # Get performance summary
        summary = self.performance_monitor.get_summary()
        
        # Calculate averages
        avg_forward_time = np.mean(forward_times) * 1000  # Convert to milliseconds
        std_forward_time = np.std(forward_times) * 1000
        avg_backward_time = np.mean(backward_times) * 1000  # Convert to milliseconds
        std_backward_time = np.std(backward_times) * 1000
        
        results = {
            'model_name': model_name,
            'profile': profile,
            'performance': {
                'avg_forward_time_ms': avg_forward_time,
                'std_forward_time_ms': std_forward_time,
                'avg_backward_time_ms': avg_backward_time,
                'std_backward_time_ms': std_backward_time,
                'total_memory_mb': summary.get('end_memory_mb', 0),
                'memory_increase_mb': summary.get('memory_increase_mb', 0),
                'avg_gradient_norm': summary.get('avg_gradient_norm', 0),
                'max_gradient_norm': summary.get('max_gradient_norm', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results to debugger if available
        if self.debugger is not None:
            self.debugger.log_step("benchmark_complete", results)
        
        self.comparison_results.append(results)
        return results
    
    def compare_models(self, models: Dict[str, Tuple[nn.Module, Tuple[int, ...]]], 
                      num_iterations: int = 10) -> List[Dict[str, Any]]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary mapping model names to (model, input_shape) tuples
            num_iterations: Number of iterations for averaging
            
        Returns:
            List of comparison results
        """
        print("Running comparative performance analysis...")
        print("="*60)
        
        results = []
        
        for model_name, (model, input_shape) in models.items():
            try:
                result = self.benchmark_model(model, model_name, input_shape, num_iterations)
                results.append(result)
                print(f"✅ {model_name} benchmark completed")
            except Exception as e:
                print(f"❌ Error benchmarking {model_name}: {e}")
                results.append({
                    'model_name': model_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        self.comparison_results = results
        return results
    
    def generate_performance_table(self) -> str:
        """
        Generate comparative performance table.
        
        Returns:
            Markdown table as string
        """
        if not self.comparison_results:
            return "No comparison results available."
        
        # Table header
        table = "| Model | Parameters | FLOPs | Forward Time (ms) | Backward Time (ms) | Memory (MB) | Grad Norm |\n"
        table += "|-------|------------|-------|-------------------|--------------------|-------------|-----------|\n"
        
        # Table rows
        for result in self.comparison_results:
            if 'error' in result:
                table += f"| {result['model_name']} | Error | Error | Error | Error | Error | Error |\n"
                continue
            
            model_name = result['model_name']
            profile = result['profile']
            performance = result['performance']
            
            # Format values
            params = f"{profile['total_parameters']:,}"
            flops = f"{profile['estimated_flops']:,}"
            forward_time = f"{performance['avg_forward_time_ms']:.2f}±{performance['std_forward_time_ms']:.2f}"
            backward_time = f"{performance['avg_backward_time_ms']:.2f}±{performance['std_backward_time_ms']:.2f}"
            memory = f"{performance['total_memory_mb']:.1f}"
            grad_norm = f"{performance['avg_gradient_norm']:.4f}"
            
            table += f"| {model_name} | {params} | {flops} | {forward_time} | {backward_time} | {memory} | {grad_norm} |\n"
        
        return table
    
    def save_comparison_results(self, filename: str = None) -> str:
        """
        Save comparison results to file.
        
        Args:
            filename: Filename to save to (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"comparison_results_{timestamp}.json"
        else:
            filename = Path(filename)
        
        # Add metadata
        results_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'results': self.comparison_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        return str(filename)
    
    def print_comparison_summary(self):
        """
        Print comparison summary.
        """
        print("\n" + "="*80)
        print("COMPARATIVE PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        table = self.generate_performance_table()
        print(table)
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total models compared: {len(self.comparison_results)}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Comparative Performance Analyzer...")
    
    # Create simple models for testing
    class SimpleModelA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    class SimpleModelB(nn.Module):
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
    
    # Test analyzer
    analyzer = ComparativePerformanceAnalyzer("test_analysis_logs")
    
    # Define models to compare
    models_to_compare = {
        "SimpleModelA": (SimpleModelA(), (32, 128)),
        "SimpleModelB": (SimpleModelB(), (32, 128))
    }
    
    # Run comparison
    results = analyzer.compare_models(models_to_compare, num_iterations=5)
    
    # Print summary
    analyzer.print_comparison_summary()
    
    # Save results
    results_file = analyzer.save_comparison_results()
    print(f"\nResults saved to: {results_file}")
    
    print("✅ Comparative performance analyzer test completed!")