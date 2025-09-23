"""
Performance Measurement Tools for Neural State Machine Models

This module provides tools for measuring training time, memory usage, and gradient flow.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn


class PerformanceMonitor:
    """
    Monitor training performance including time, memory, and gradient flow.
    """

    def __init__(self, log_dir: str = "performance_logs"):
        """
        Initialize performance monitor.

        Args:
            log_dir: Directory to save performance logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.start_time = None
        self.start_memory = None
        self.memory_snapshots = []
        self.gradient_stats = []
        self.forward_times = []
        self.backward_times = []

        # Get process for memory monitoring
        self.process = psutil.Process(os.getpid())

    def start_monitoring(self):
        """
        Start performance monitoring.
        """
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_snapshots = []
        self.gradient_stats = []
        self.forward_times = []
        self.backward_times = []

        print(f"Started performance monitoring at {datetime.now().isoformat()}")

    def record_memory_snapshot(self, label: str = ""):
        """
        Record memory usage snapshot.

        Args:
            label: Label for this snapshot
        """
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        timestamp = time.time() - self.start_time if self.start_time else 0

        snapshot = {"timestamp": timestamp, "memory_mb": current_memory, "label": label}

        self.memory_snapshots.append(snapshot)
        return snapshot

    def measure_forward_pass(
        self, model: nn.Module, inputs: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Measure forward pass performance.

        Args:
            model: Model to test
            inputs: Input tensor

        Returns:
            Dictionary with forward pass metrics
        """
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Record start
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)

        # Record end
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024

        forward_time = end_time - start_time
        memory_delta = end_memory - start_memory

        self.forward_times.append(forward_time)

        metrics = {
            "forward_time": forward_time,
            "memory_delta_mb": memory_delta,
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape) if hasattr(outputs, "shape") else "N/A",
        }

        return metrics

    def measure_backward_pass(
        self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None
    ) -> Dict[str, Any]:
        """
        Measure backward pass performance.

        Args:
            model: Model to test
            inputs: Input tensor
            targets: Target tensor (optional)

        Returns:
            Dictionary with backward pass metrics
        """
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Zero gradients
        model.zero_grad()

        # Record start
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024

        # Forward pass
        outputs = model(inputs)

        # Create loss if targets provided
        if targets is not None:
            if outputs.size(-1) != targets.size(-1):
                # Adjust dimensions
                if len(outputs.shape) > len(targets.shape):
                    targets = targets.unsqueeze(-1)
                criterion = nn.MSELoss()
            else:
                criterion = nn.MSELoss()
            loss = criterion(outputs, targets)
        else:
            # Use sum of outputs as loss
            loss = outputs.sum()

        # Backward pass
        loss.backward()

        # Record end
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024

        backward_time = end_time - start_time
        memory_delta = end_memory - start_memory

        self.backward_times.append(backward_time)

        # Collect gradient statistics
        grad_stats = self._collect_gradient_statistics(model)
        self.gradient_stats.append(grad_stats)

        metrics = {
            "backward_time": backward_time,
            "memory_delta_mb": memory_delta,
            "loss": loss.item(),
            "gradient_stats": grad_stats,
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape) if hasattr(outputs, "shape") else "N/A",
        }

        return metrics

    def _collect_gradient_statistics(self, model: nn.Module) -> Dict[str, float]:
        """
        Collect gradient statistics from model parameters.

        Args:
            model: Model to analyze

        Returns:
            Dictionary with gradient statistics
        """
        grad_norms = []
        grad_means = []
        grad_stds = []

        for param in model.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_mean = torch.mean(param.grad).item()
                grad_std = torch.std(param.grad).item()

                grad_norms.append(grad_norm)
                grad_means.append(grad_mean)
                grad_stds.append(grad_std)

        if grad_norms:
            return {
                "mean_norm": np.mean(grad_norms),
                "max_norm": np.max(grad_norms),
                "min_norm": np.min(grad_norms),
                "std_norm": np.std(grad_norms),
                "mean_gradient": np.mean(grad_means),
                "std_gradient": np.mean(grad_stds),
                "total_parameters": len(grad_norms),
            }
        else:
            return {
                "mean_norm": 0.0,
                "max_norm": 0.0,
                "min_norm": 0.0,
                "std_norm": 0.0,
                "mean_gradient": 0.0,
                "std_gradient": 0.0,
                "total_parameters": 0,
            }

    def measure_training_epoch(
        self, model: nn.Module, dataloader, optimizer, criterion, device: torch.device
    ) -> Dict[str, Any]:
        """
        Measure performance of a training epoch.

        Args:
            model: Model to train
            dataloader: DataLoader with training data
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use

        Returns:
            Dictionary with epoch metrics
        """
        model.train()
        epoch_start = time.time()

        total_loss = 0.0
        batch_times = []
        memory_usage = []

        for batch_idx, (data, target) in enumerate(dataloader):
            batch_start = time.time()

            data, target = data.to(device), target.to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Adjust output/target dimensions if needed
            if output.size(-1) != target.size(-1):
                if len(output.shape) > len(target.shape):
                    target = target.unsqueeze(-1).expand_as(output)
                elif len(target.shape) > len(output.shape):
                    output = output.unsqueeze(-1).expand_as(target)

            loss = criterion(output, target)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Record memory usage
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_usage.append(current_memory)

            # Limit for demo
            if batch_idx > 10:
                break

        epoch_time = time.time() - epoch_start

        metrics = {
            "epoch_time": epoch_time,
            "avg_batch_time": np.mean(batch_times),
            "total_loss": total_loss,
            "avg_loss": total_loss / len(batch_times),
            "avg_memory_mb": np.mean(memory_usage),
            "max_memory_mb": np.max(memory_usage),
            "batches_processed": len(batch_times),
        }

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.

        Returns:
            Dictionary with performance summary
        """
        if not self.start_time:
            return {"error": "Monitoring not started"}

        total_time = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / 1024 / 1024
        memory_increase = end_memory - self.start_memory if self.start_memory else 0

        summary = {
            "total_time_seconds": total_time,
            "start_memory_mb": self.start_memory,
            "end_memory_mb": end_memory,
            "memory_increase_mb": memory_increase,
            "memory_snapshots": self.memory_snapshots,
            "forward_passes": len(self.forward_times),
            "avg_forward_time": (
                np.mean(self.forward_times) if self.forward_times else 0
            ),
            "backward_passes": len(self.backward_times),
            "avg_backward_time": (
                np.mean(self.backward_times) if self.backward_times else 0
            ),
            "gradient_measurements": len(self.gradient_stats),
        }

        # Add gradient statistics if available
        if self.gradient_stats:
            avg_grad_norm = np.mean([s["mean_norm"] for s in self.gradient_stats])
            max_grad_norm = np.max([s["max_norm"] for s in self.gradient_stats])
            summary["avg_gradient_norm"] = avg_grad_norm
            summary["max_gradient_norm"] = max_grad_norm

        return summary

    def save_summary(self, filename: str = None) -> str:
        """
        Save performance summary to file.

        Args:
            filename: Filename to save to (optional)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"performance_summary_{timestamp}.json"
        else:
            filename = Path(filename)

        summary = self.get_summary()
        summary["timestamp"] = datetime.now().isoformat()

        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)

        return str(filename)


class ModelProfiler:
    """
    Profile model performance including FLOPs estimation.
    """

    def __init__(self):
        """Initialize model profiler."""
        pass

    def estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """
        Estimate FLOPs for a model (simplified estimation).

        Args:
            model: Model to profile
            input_shape: Input tensor shape

        Returns:
            Estimated FLOPs
        """
        total_flops = 0

        # Create sample input
        sample_input = torch.randn(input_shape)

        # Count FLOPs for each layer (simplified)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear layer: input_features * output_features * 2 (multiply + add)
                flops = module.in_features * module.out_features * 2
                total_flops += flops
            elif isinstance(module, nn.Conv2d):
                # Conv2d: output_h * output_w * out_channels * in_channels * kernel_h * kernel_w
                flops = (
                    module.out_channels
                    * module.in_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                )
                # Approximate output size
                output_size = (
                    sample_input.shape[-1] // module.stride[0]
                    if module.stride[0] > 0
                    else sample_input.shape[-1]
                )
                flops *= output_size * output_size
                total_flops += flops
            elif isinstance(module, nn.LSTM):
                # LSTM: 4 * (input_size + hidden_size) * hidden_size * seq_len
                if hasattr(module, "input_size") and hasattr(module, "hidden_size"):
                    flops = (
                        4
                        * (module.input_size + module.hidden_size)
                        * module.hidden_size
                    )
                    # Approximate sequence length
                    seq_len = (
                        sample_input.shape[1] if len(sample_input.shape) > 1 else 10
                    )
                    total_flops += flops * seq_len

        return total_flops

    def profile_model(
        self, model: nn.Module, input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Profile model characteristics.

        Args:
            model: Model to profile
            input_shape: Input tensor shape

        Returns:
            Dictionary with model profile
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate FLOPs
        flops = self.estimate_flops(model, input_shape)

        # Model size
        model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)

        profile = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_flops": flops,
            "model_size_mb": model_size_mb,
            "input_shape": input_shape,
        }

        return profile


# Example usage
if __name__ == "__main__":
    print("Testing Performance Measurement Tools...")

    # Create simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # Test performance monitor
    monitor = PerformanceMonitor("test_performance_logs")
    monitor.start_monitoring()

    # Test model
    model = TestModel()
    inputs = torch.randn(32, 128)
    targets = torch.randn(32, 128)

    # Measure forward pass
    forward_metrics = monitor.measure_forward_pass(model, inputs)
    print(f"Forward pass metrics: {forward_metrics}")

    # Measure backward pass
    backward_metrics = monitor.measure_backward_pass(model, inputs, targets)
    print(f"Backward pass metrics: {backward_metrics}")

    # Get summary
    summary = monitor.get_summary()
    print(f"Performance summary: {summary}")

    # Save summary
    log_file = monitor.save_summary()
    print(f"Summary saved to: {log_file}")

    # Test model profiler
    profiler = ModelProfiler()
    profile = profiler.profile_model(model, (32, 128))
    print(f"Model profile: {profile}")

    print("✅ Performance measurement tools test completed!")
