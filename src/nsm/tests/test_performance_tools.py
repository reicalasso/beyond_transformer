"""
Test script for performance measurement tools
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# Test imports
try:
    # Try absolute imports first
    from nsm.utils.performance_monitor import PerformanceMonitor, ModelProfiler
    from nsm.utils.comparative_analyzer import ComparativePerformanceAnalyzer
except ImportError:
    # Fall back to relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.performance_monitor import PerformanceMonitor, ModelProfiler
    from utils.comparative_analyzer import ComparativePerformanceAnalyzer
    
    print("✅ All performance measurement imports successful")
    
    # Test PerformanceMonitor
    monitor = PerformanceMonitor("test_perf_logs")
    monitor.start_monitoring()
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.fc(x))
    
    model = SimpleModel()
    inputs = torch.randn(16, 64)
    
    # Test forward pass measurement
    forward_metrics = monitor.measure_forward_pass(model, inputs)
    print(f"✅ Forward pass measurement: {forward_metrics['forward_time']:.4f}s")
    
    # Test ModelProfiler
    profiler = ModelProfiler()
    profile = profiler.profile_model(model, (16, 64))
    print(f"✅ Model profiling: {profile['total_parameters']:,} parameters")
    
    # Test ComparativePerformanceAnalyzer
    analyzer = ComparativePerformanceAnalyzer("test_comp_logs")
    
    models = {
        "SimpleModel": (model, (16, 64))
    }
    
    results = analyzer.compare_models(models, num_iterations=2)
    print(f"✅ Comparative analysis: {len(results)} models analyzed")
    
    # Generate performance table
    table = analyzer.generate_performance_table()
    print("✅ Performance table generated")
    
    print("\n🎉 All performance measurement tools working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()