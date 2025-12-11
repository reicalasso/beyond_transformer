"""
Comprehensive Performance Measurement Script

This script measures training time, memory usage, and gradient flow for PULSE models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any

from pulse.utils.comparative_analyzer import ComparativePerformanceAnalyzer
from pulse.models import SimplePulse, AdvancedHybridModel, SequentialHybridModel


class PerformanceMeasurementSuite:
    """
    Comprehensive performance measurement suite.
    """
    
    def __init__(self, log_dir: str = "performance_measurements"):
        """
        Initialize performance measurement suite.
        
        Args:
            log_dir: Directory to save measurement results
        """
        self.log_dir = log_dir
        self.analyzer = ComparativePerformanceAnalyzer(log_dir)
    
    def create_test_models(self) -> Dict[str, tuple]:
        """
        Create test models for performance measurement.
        
        Returns:
            Dictionary mapping model names to (model, input_shape) tuples
        """
        models = {}
        
        # SimplePulse
        try:
            simple_pulse = SimplePulse(
                input_dim=784,
                state_dim=128,
                num_states=16,
                output_dim=10,
                gate_type='gru'
            )
            models["SimplePulse"] = (simple_pulse, (32, 784))
        except Exception as e:
            print(f"Warning: Could not create SimplePulse: {e}")
        
        # AdvancedHybridModel
        try:
            advanced_config = {
                'input_dim': 784,
                'output_dim': 10,
                'embedding_dim': 64,
                'sequence_length': 8,
                'ssm_dim': 64,
                'ntm_mem_size': 32,
                'ntm_mem_dim': 16,
                'rnn_hidden_dim': 64,
                'attention_heads': 4
            }
            advanced_hybrid = AdvancedHybridModel(advanced_config)
            models["AdvancedHybrid"] = (advanced_hybrid, (32, 784))
        except Exception as e:
            print(f"Warning: Could not create AdvancedHybridModel: {e}")
        
        # SequentialHybridModel
        try:
            sequential_hybrid = SequentialHybridModel(input_dim=784, output_dim=10)
            models["SequentialHybrid"] = (sequential_hybrid, (32, 784))
        except Exception as e:
            print(f"Warning: Could not create SequentialHybridModel: {e}")
        
        # Baseline models for comparison
        try:
            # Simple LSTM
            lstm_model = nn.LSTM(64, 128, batch_first=True)
            models["LSTM"] = (lstm_model, (32, 10, 64))  # [batch, seq, features]
        except Exception as e:
            print(f"Warning: Could not create LSTM: {e}")
        
        try:
            # Simple Transformer-like model
            transformer_model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
                num_layers=2
            )
            models["Transformer"] = (transformer_model, (32, 10, 64))  # [batch, seq, features]
        except Exception as e:
            print(f"Warning: Could not create Transformer: {e}")
        
        return models
    
    def measure_training_performance(self, model: nn.Module, model_name: str,
                                   input_shape: tuple, epochs: int = 3) -> Dict[str, Any]:
        """
        Measure training performance.
        
        Args:
            model: Model to train
            model_name: Name of the model
            input_shape: Input tensor shape
            epochs: Number of epochs to train
            
        Returns:
            Dictionary with training performance metrics
        """
        print(f"Measuring training performance for {model_name}...")
        
        # Create dummy dataset
        batch_size = input_shape[0]
        dataset_size = batch_size * 20  # 20 batches
        
        # Generate dummy data
        if len(input_shape) == 2:
            X = torch.randn(dataset_size, input_shape[1])
            y = torch.randint(0, 10, (dataset_size,))  # Classification targets
        elif len(input_shape) == 3:
            X = torch.randn(dataset_size, input_shape[1], input_shape[2])
            y = torch.randint(0, 10, (dataset_size,))
        else:
            # Fallback for other shapes
            X = torch.randn(dataset_size, 64)
            y = torch.randint(0, 10, (dataset_size,))
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Measure training performance
        self.analyzer.performance_monitor.start_monitoring()
        
        epoch_times = []
        memory_usage = []
        
        for epoch in range(epochs):
            epoch_metrics = self.analyzer.performance_monitor.measure_training_epoch(
                model, dataloader, optimizer, criterion, device)
            epoch_times.append(epoch_metrics['epoch_time'])
            memory_usage.append(epoch_metrics['max_memory_mb'])
        
        # Get final summary
        summary = self.analyzer.performance_monitor.get_summary()
        
        training_metrics = {
            'model_name': model_name,
            'avg_epoch_time': np.mean(epoch_times),
            'total_training_time': np.sum(epoch_times),
            'avg_memory_mb': np.mean(memory_usage),
            'max_memory_mb': np.max(memory_usage),
            'epochs': epochs,
            'performance_summary': summary
        }
        
        return training_metrics
    
    def run_comprehensive_analysis(self, num_iterations: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive performance analysis.
        
        Args:
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print("Running Comprehensive Performance Analysis")
        print("="*60)
        
        # Create models
        models = self.create_test_models()
        print(f"Created {len(models)} models for analysis")
        
        if not models:
            print("No models available for analysis")
            return {}
        
        # Run comparative benchmarking
        print("\n1. Running comparative benchmarking...")
        benchmark_results = self.analyzer.compare_models(models, num_iterations)
        
        # Run training performance measurement for selected models
        print("\n2. Measuring training performance...")
        training_results = []
        
        # Select a few models for detailed training analysis
        selected_models = {k: v for k, v in list(models.items())[:3]}  # First 3 models
        
        for model_name, (model, input_shape) in selected_models.items():
            try:
                training_metrics = self.measure_training_performance(
                    model, model_name, input_shape, epochs=2)  # Fewer epochs for demo
                training_results.append(training_metrics)
                print(f"✅ Training performance measured for {model_name}")
            except Exception as e:
                print(f"❌ Error measuring training performance for {model_name}: {e}")
                training_results.append({
                    'model_name': model_name,
                    'error': str(e)
                })
        
        # Compile results
        results = {
            'timestamp': torch.datetime.now().isoformat(),
            'benchmark_results': benchmark_results,
            'training_results': training_results,
            'models_tested': list(models.keys())
        }
        
        return results
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """
        Generate detailed performance report.
        
        Args:
            results: Analysis results
            
        Returns:
            Detailed report as string
        """
        if not results:
            return "No results available for report generation."
        
        report = []
        report.append("COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Generated at: {results.get('timestamp', 'N/A')}")
        report.append(f"Models tested: {', '.join(results.get('models_tested', []))}")
        report.append("")
        
        # Benchmark results section
        report.append("1. BENCHMARK RESULTS")
        report.append("-"*30)
        benchmark_table = self.analyzer.generate_performance_table()
        report.append(benchmark_table)
        report.append("")
        
        # Training performance section
        report.append("2. TRAINING PERFORMANCE")
        report.append("-"*30)
        training_results = results.get('training_results', [])
        
        if training_results:
            report.append("| Model | Avg Epoch Time (s) | Total Time (s) | Max Memory (MB) |")
            report.append("|-------|-------------------|----------------|-----------------|")
            
            for result in training_results:
                if 'error' in result:
                    report.append(f"| {result['model_name']} | Error | Error | Error |")
                    continue
                
                model_name = result['model_name']
                avg_epoch = result['avg_epoch_time']
                total_time = result['total_training_time']
                max_memory = result['max_memory_mb']
                
                report.append(f"| {model_name} | {avg_epoch:.2f} | {total_time:.2f} | {max_memory:.1f} |")
        else:
            report.append("No training performance data available.")
        
        report.append("")
        report.append("3. ANALYSIS SUMMARY")
        report.append("-"*30)
        report.append("This report provides comparative performance metrics for")
        report.append("PULSE models and baseline architectures.")
        report.append("Metrics include parameter count, FLOPs estimation,")
        report.append("forward/backward pass times, memory usage, and gradient norms.")
        
        return "\n".join(report)
    
    def save_detailed_report(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save detailed performance report.
        
        Args:
            results: Analysis results
            filename: Filename to save to (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(self.log_dir) / f"detailed_report_{timestamp}.txt"
        else:
            filename = Path(filename)
        
        report = self.generate_detailed_report(results)
        
        with open(filename, 'w') as f:
            f.write(report)
        
        return str(filename)


# Example usage
if __name__ == "__main__":
    print("PULSE - Performance Measurement Suite")
    print("="*60)
    
    # Create measurement suite
    suite = PerformanceMeasurementSuite("performance_logs")
    
    # Run comprehensive analysis
    print("Starting performance analysis...")
    results = suite.run_comprehensive_analysis(num_iterations=3)  # Fewer iterations for demo
    
    # Print summary
    suite.analyzer.print_comparison_summary()
    
    # Generate and save detailed report
    report_file = suite.save_detailed_report(results)
    print(f"\nDetailed report saved to: {report_file}")
    
    # Also save comparison results
    comparison_file = suite.analyzer.save_comparison_results()
    print(f"Comparison results saved to: {comparison_file}")
    
    print("\n🎉 Performance measurement suite completed!")