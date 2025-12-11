"""
Unit tests for debugger and performance monitoring tools.
"""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from pulse.utils.comparative_analyzer import ComparativePerformanceAnalyzer
from pulse.utils.debugger import PulseDebugger
from pulse.utils.performance_monitor import PerformanceMonitor


class TestPulseDebugger:
    """Test suite for PulseDebugger."""

    @pytest.fixture
    def debugger(self, temp_dir):
        """Create a PulseDebugger for testing."""
        return PulseDebugger(log_dir=str(temp_dir / "debug_logs"), verbose=False)

    def test_initialization(self, temp_dir):
        """Test PulseDebugger initialization."""
        debugger = PulseDebugger(log_dir=str(temp_dir / "test_logs"), verbose=True)

        assert debugger.log_dir.exists()
        assert debugger.verbose is True
        assert debugger.debug_enabled is False
        assert len(debugger.log_data) == 0

    def test_enable_disable_debug(self, debugger):
        """Test enabling and disabling debug mode."""
        assert debugger.debug_enabled is False

        # Enable debug
        debugger.enable_debug()
        assert debugger.debug_enabled is True
        assert debugger.step_counter == 0
        assert len(debugger.log_data) == 0

        # Disable debug
        debugger.disable_debug()
        assert debugger.debug_enabled is False

    def test_log_step(self, debugger):
        """Test logging a step."""
        debugger.enable_debug()

        test_data = {
            "input_tensor": torch.randn(4, 8),
            "output_tensor": torch.randn(4, 8),
            "processing_info": {"layer": "test", "operation": "forward"},
        }

        debugger.log_step("test_step", test_data, {"batch_size": 4})

        assert len(debugger.log_data) == 1
        assert debugger.step_counter == 1
        assert debugger.log_data[0]["step_name"] == "test_step"
        assert "timestamp" in debugger.log_data[0]

    def test_log_memory_operation(self, debugger):
        """Test logging memory operations."""
        debugger.enable_debug()

        read_data = torch.randn(8)
        attention_weights = torch.softmax(torch.randn(8), dim=0)

        debugger.log_memory_operation(
            "read",
            "memory_slot_0",
            read_data=read_data,
            attention_weights=attention_weights,
            operation_info={"purpose": "retrieve_context"},
        )

        assert len(debugger.log_data) == 1
        log_entry = debugger.log_data[0]
        assert "memory_operation" in log_entry["data"]

    def test_log_attention_operation(self, debugger):
        """Test logging attention operations."""
        debugger.enable_debug()

        attention_weights = torch.softmax(torch.randn(8, 8), dim=-1)
        attended_values = torch.randn(8, 8)

        debugger.log_attention_operation(
            "token_to_state",
            "tokens",
            "states",
            attention_weights=attention_weights,
            attended_values=attended_values,
            operation_info={"layer": "encoder"},
        )

        assert len(debugger.log_data) == 1
        log_entry = debugger.log_data[0]
        assert "attention_operation" in log_entry["data"]

    def test_log_state_update(self, debugger):
        """Test logging state updates."""
        debugger.enable_debug()

        old_state = torch.randn(6, 12)
        new_state = old_state + torch.randn(6, 12) * 0.1

        debugger.log_state_update(
            "test_component",
            old_state,
            new_state,
            update_info={"update_type": "propagation"},
        )

        assert len(debugger.log_data) == 1
        log_entry = debugger.log_data[0]
        assert "state_update" in log_entry["data"]

    def test_save_debug_log(self, debugger, temp_dir):
        """Test saving debug log to file."""
        debugger.enable_debug()

        # Add some log entries
        debugger.log_step("test_step_1", {"data": torch.randn(4, 8)})
        debugger.log_step("test_step_2", {"data": torch.randn(4, 8)})

        # Save log
        log_file = debugger.save_debug_log()

        assert os.path.exists(log_file)

        # Check log content
        with open(log_file, "r") as f:
            log_data = json.load(f)

        assert "log_data" in log_data
        assert len(log_data["log_data"]) == 2
        assert log_data["total_steps"] == 2

    def test_get_step_summary(self, debugger):
        """Test getting step summary."""
        debugger.enable_debug()

        # Add different types of steps
        debugger.log_step("step_1", {"data": torch.randn(4, 8)})
        debugger.log_step("step_1", {"data": torch.randn(4, 8)})
        debugger.log_step("step_2", {"data": torch.randn(4, 8)})
        debugger.log_memory_operation("read", "slot_0")
        debugger.log_memory_operation("write", "slot_1")

        summary = debugger.get_step_summary()

        assert "step_1" in summary
        assert "step_2" in summary
        assert summary["step_1"] == 2
        assert summary["step_2"] == 1

    def test_print_summary(self, debugger, capsys, caplog):
        """Test printing debug summary."""
        import logging

        caplog.set_level(logging.INFO)

        debugger.enable_debug()

        # Add some entries
        debugger.log_step("test_step", {"data": torch.randn(4, 8)})
        debugger.log_memory_operation("read", "slot_0")

        debugger.print_summary()

        # Should contain summary information in logs
        assert "PULSE Debug Summary" in caplog.text
        assert "Total steps logged:" in caplog.text

    def test_serialize_tensor(self, debugger):
        """Test tensor serialization."""
        tensor = torch.randn(4, 8)
        serialized = debugger._serialize_tensor(tensor)

        assert isinstance(serialized, dict)
        assert "shape" in serialized
        assert "mean" in serialized
        assert "std" in serialized
        assert "min" in serialized
        assert "max" in serialized
        assert "sample" in serialized

    def test_serialize_data(self, debugger):
        """Test data serialization."""
        test_data = {
            "tensor_data": torch.randn(4, 8),
            "dict_data": {"nested": torch.randn(2, 4)},
            "list_data": [torch.randn(4) for _ in range(3)],
            "scalar_data": 42,
        }

        serialized = debugger._serialize_data(test_data)

        assert isinstance(serialized, dict)
        assert "tensor_data" in serialized
        assert "dict_data" in serialized
        assert "list_data" in serialized
        assert "scalar_data" in serialized


class TestPerformanceMonitor:
    """Test suite for PerformanceMonitor."""

    @pytest.fixture
    def perf_monitor(self):
        """Create a PerformanceMonitor for testing."""
        return PerformanceMonitor()

    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_measure_forward_pass(self, perf_monitor):
        """Test measuring forward pass."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(4, 16)

        metrics = perf_monitor.measure_forward_pass(model, inputs)

        assert "forward_time" in metrics
        assert "memory_delta_mb" in metrics
        assert metrics["forward_time"] > 0

    def test_measure_backward_pass(self, perf_monitor):
        """Test measuring backward pass."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(4, 16)
        targets = torch.randn(4, 8)

        metrics = perf_monitor.measure_backward_pass(model, inputs, targets)

        assert "backward_time" in metrics
        assert "memory_delta_mb" in metrics
        assert "loss" in metrics
        assert metrics["backward_time"] > 0

    def test_collect_gradient_statistics(self, perf_monitor):
        """Test collecting gradient statistics."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(4, 16)
        targets = torch.randn(4, 8)

        # Create some gradients
        output = model(inputs)
        loss = torch.nn.functional.mse_loss(output, targets)
        loss.backward()

        grad_stats = perf_monitor._collect_gradient_statistics(model)

        assert "mean_norm" in grad_stats
        assert "max_norm" in grad_stats
        assert "total_parameters" in grad_stats
        assert grad_stats["total_parameters"] > 0


class TestComparativePerformanceAnalyzer:
    """Test suite for ComparativePerformanceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a ComparativePerformanceAnalyzer for testing."""
        return ComparativePerformanceAnalyzer()

    def test_initialization(self):
        """Test ComparativePerformanceAnalyzer initialization."""
        analyzer = ComparativePerformanceAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "performance_monitor")
        assert hasattr(analyzer, "model_profiler")

    def test_benchmark_model(self, analyzer):
        """Test benchmarking a model."""

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        results = analyzer.benchmark_model(
            model, "SimpleModel", (4, 16), num_iterations=2
        )

        assert "model_name" in results
        assert "profile" in results
        assert "performance" in results
        assert results["model_name"] == "SimpleModel"

    def test_compare_models(self, analyzer):
        """Test comparing multiple models."""

        class ModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        class ModelB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 32)
                self.linear2 = torch.nn.Linear(32, 8)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                return self.linear2(x)

        models = {"ModelA": (ModelA(), (4, 16)), "ModelB": (ModelB(), (4, 16))}

        results = analyzer.compare_models(models, num_iterations=2)

        assert len(results) == 2
        for result in results:
            assert "model_name" in result
            assert "profile" in result
            assert "performance" in result


# Integration tests
class TestDebugAndPerformanceIntegration:
    """Integration tests for debug and performance tools."""

    @pytest.mark.integration
    def test_debug_performance_workflow(self, temp_dir):
        """Test complete debug and performance workflow."""
        # Create tools
        debugger = PulseDebugger(log_dir=str(temp_dir / "debug_logs"), verbose=False)
        perf_monitor = PerformanceMonitor()
        analyzer = ComparativePerformanceAnalyzer()

        # Enable debug
        debugger.enable_debug()

        # Create test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(32, 64)
                self.linear2 = torch.nn.Linear(64, 32)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                return self.linear2(x)

        model = TestModel()
        inputs = torch.randn(8, 32)

        # Debug and performance monitoring
        debugger.log_step("forward_start", {"input_shape": inputs.shape})

        # Measure performance
        perf_metrics = perf_monitor.measure_forward_pass(model, inputs)
        debugger.log_step("forward_complete", perf_metrics)

        # Log some operations
        debugger.log_memory_operation(
            "read",
            "test_slot",
            read_data=torch.randn(16),
            attention_weights=torch.softmax(torch.randn(16), dim=0),
        )

        debugger.log_attention_operation(
            "self_attention",
            "query",
            "key",
            attention_weights=torch.softmax(torch.randn(8, 8), dim=-1),
        )

        # Save logs
        debug_log = debugger.save_debug_log()
        assert os.path.exists(debug_log)

        # Check log content
        with open(debug_log, "r") as f:
            log_data = json.load(f)

        assert len(log_data["log_data"]) >= 3

    @pytest.mark.integration
    def test_model_comparison_with_debugging(self, temp_dir):
        """Test model comparison with debugging enabled."""
        # Create debugger
        debugger = PulseDebugger(log_dir=str(temp_dir / "comparison_logs"), verbose=False)
        debugger.enable_debug()

        # Create analyzer
        analyzer = ComparativePerformanceAnalyzer()
        # Set debugger on analyzer
        analyzer.set_debugger(debugger)

        # Create simple models
        class SimpleModelA(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        class SimpleModelB(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(16, 32)
                self.linear2 = torch.nn.Linear(32, 8)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                return self.linear2(x)

        models = {
            "SimpleA": (SimpleModelA(), (4, 16)),
            "SimpleB": (SimpleModelB(), (4, 16)),
        }

        # Compare models with debugging
        results = analyzer.compare_models(models, num_iterations=2)

        assert len(results) == 2
        for result in results:
            assert "model_name" in result
            assert "profile" in result

        # Check that debug logs were created
        assert len(debugger.log_data) > 0

    @pytest.mark.slow
    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        # This would typically compare against baseline performance
        # For now, we just test that the analyzer works
        analyzer = ComparativePerformanceAnalyzer()

        class RegressionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(64, 128),
                        torch.nn.ReLU(),
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 32),
                    ]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = RegressionModel()
        input_shape = (8, 64)

        # Benchmark the model
        results = analyzer.benchmark_model(
            model, "RegressionModel", input_shape, num_iterations=3
        )

        # Check results structure
        assert "model_name" in results
        assert "profile" in results
        assert "performance" in results

        performance = results["performance"]
        assert "avg_forward_time_ms" in performance
        assert "avg_backward_time_ms" in performance
        assert performance["avg_forward_time_ms"] > 0
        assert performance["avg_backward_time_ms"] > 0


# Performance tests
class TestDebugPerformance:
    """Performance tests for debug and monitoring tools."""

    @pytest.mark.slow
    def test_debug_overhead_minimization(self, temp_dir):
        """Test that debugging adds minimal overhead."""
        import time

        # Without debugging
        debugger_off = PulseDebugger(log_dir=str(temp_dir / "no_debug"), verbose=False)

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 32)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(16, 64)

        # Time without debugging
        start_time = time.time()
        for _ in range(100):
            _ = model(inputs)
        time_without_debug = time.time() - start_time

        # With debugging (but disabled)
        debugger_off.enable_debug()
        debugger_off.disable_debug()

        start_time = time.time()
        for _ in range(100):
            _ = model(inputs)
            # Minimal debug logging (should be negligible)
            debugger_off.log_step("minimal", {"dummy": 1})
        time_with_minimal_debug = time.time() - start_time

        # Overhead should be minimal (< 10%)
        overhead_ratio = time_with_minimal_debug / time_without_debug
        assert overhead_ratio < 1.1, f"Debug overhead too high: {overhead_ratio:.3f}x"

    def test_logging_performance(self, temp_dir):
        """Test logging performance."""
        import time

        debugger = PulseDebugger(log_dir=str(temp_dir / "perf_test"), verbose=False)
        debugger.enable_debug()

        # Test logging many entries
        test_data = {"tensor": torch.randn(32, 64)}

        start_time = time.time()
        for i in range(1000):
            debugger.log_step(f"step_{i}", test_data)
        log_time = time.time() - start_time

        # Should be reasonably fast (less than 1 second for 1000 entries)
        assert log_time < 1.0, f"Logging too slow: {log_time:.4f}s for 1000 entries"

        # Check log size
        assert len(debugger.log_data) == 1000

    @pytest.mark.gpu
    def test_gpu_memory_monitoring(self):
        """Test GPU memory monitoring."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        debugger = PulseDebugger(verbose=False)
        debugger.enable_debug()

        # Allocate some GPU memory
        gpu_tensor = torch.randn(1000, 1000).cuda()

        # Log memory snapshot
        snapshot = debugger.record_memory_snapshot("gpu_allocation")

        assert "memory_mb" in snapshot
        assert snapshot["memory_mb"] > 0

        # Clean up
        del gpu_tensor
        torch.cuda.empty_cache()


# Test utility functions
class TestDebugUtilities:
    """Test utility functions for debugging."""

    def test_logger_configuration(self, temp_dir):
        """Test logger configuration."""
        import logging

        debugger = PulseDebugger(log_dir=str(temp_dir / "logger_test"), verbose=True)

        # Check that logger is configured
        logger = debugger.logger
        assert logger is not None
        assert len(logger.handlers) >= 1

        # Test logging levels
        assert logger.level <= logging.DEBUG

    def test_process_monitoring(self):
        """Test process monitoring capabilities."""
        debugger = PulseDebugger(verbose=False)
        debugger.enable_debug()

        # Record initial snapshot
        initial_snapshot = debugger.record_memory_snapshot("initial")

        assert "memory_mb" in initial_snapshot
        assert "timestamp" in initial_snapshot
        assert initial_snapshot["memory_mb"] >= 0

    def test_gradient_tracking(self, temp_dir):
        """Test gradient tracking capabilities."""
        debugger = PulseDebugger(log_dir=str(temp_dir / "grad_test"), verbose=False)
        debugger.enable_debug()

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 8)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        inputs = torch.randn(4, 16)
        targets = torch.randn(4, 8)

        # Forward and backward pass
        output = model(inputs)
        loss = torch.nn.functional.mse_loss(output, targets)
        loss.backward()

        # Collect gradient statistics
        perf_monitor = PerformanceMonitor()
        grad_stats = perf_monitor._collect_gradient_statistics(model)

        assert "mean_norm" in grad_stats
        assert "max_norm" in grad_stats
        assert grad_stats["total_parameters"] > 0
