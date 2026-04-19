"""
PULSE Logging & Monitoring Utilities

Provides structured logging, metrics tracking, and optional integrations
with TensorBoard and Weights & Biases.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    tokens_per_second: float = 0.0
    grad_norm: float = 0.0
    
    # Validation metrics
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    
    # Memory metrics
    gpu_memory_used: Optional[float] = None
    gpu_memory_allocated: Optional[float] = None
    
    # Timing
    step_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsLogger:
    """
    Unified metrics logger supporting multiple backends.
    
    Supports:
    - Console logging
    - JSON file logging
    - TensorBoard (optional)
    - Weights & Biases (optional)
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_name: str = "pulse_training",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        log_interval: int = 10,
    ):
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup console logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # JSON metrics file
        self.metrics_file = self.logs_dir / "metrics.jsonl"
        
        # History for plotting
        self.history: List[Dict[str, Any]] = []
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()
        
        # Weights & Biases
        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_project, wandb_config)
        
        # Timing
        self._step_start_time = None
        self._training_start_time = time.time()
    
    def _setup_tensorboard(self):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.logs_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
            self.logger.info(f"TensorBoard logging to: {tb_dir}")
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
    
    def _setup_wandb(self, project: Optional[str], config: Optional[Dict]):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=project or "pulse",
                name=self.experiment_name,
                config=config or {},
                dir=str(self.output_dir),
            )
            self.logger.info(f"W&B logging enabled: {wandb.run.url}")
        except ImportError:
            self.logger.warning("wandb not available. Install with: pip install wandb")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
    
    def start_step(self):
        """Mark the start of a training step."""
        self._step_start_time = time.time()
    
    def log_metrics(
        self,
        metrics: Union[TrainingMetrics, Dict[str, Any]],
        step: Optional[int] = None,
    ):
        """
        Log metrics to all enabled backends.
        
        Args:
            metrics: TrainingMetrics object or dict
            step: Optional step override
        """
        if isinstance(metrics, TrainingMetrics):
            metrics_dict = metrics.to_dict()
            step = step or metrics.step
        else:
            metrics_dict = metrics
            step = step or metrics_dict.get("step", 0)
        
        # Add step time if available
        if self._step_start_time:
            metrics_dict["step_time"] = time.time() - self._step_start_time
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            metrics_dict["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            metrics_dict["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        
        # Store in history
        self.history.append({"step": step, **metrics_dict})
        
        # Write to JSON file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps({"step": step, **metrics_dict}) + "\n")
        
        # TensorBoard
        if self.tb_writer:
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.wandb_run:
            import wandb
            wandb.log(metrics_dict, step=step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text (e.g., generated samples)."""
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
        
        if self.wandb_run:
            import wandb
            wandb.log({tag: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_model_gradients(self, model: torch.nn.Module, step: int):
        """Log gradient statistics for model parameters."""
        total_norm = 0.0
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Log per-layer stats (simplified names)
                layer_name = name.split(".")[0]
                if layer_name not in grad_stats:
                    grad_stats[layer_name] = []
                grad_stats[layer_name].append(param_norm)
        
        total_norm = total_norm ** 0.5
        
        # Log total gradient norm
        self.log_metrics({"grad_norm": total_norm}, step=step)
        
        # Log per-layer average gradient norms
        if self.tb_writer:
            for layer_name, norms in grad_stats.items():
                avg_norm = sum(norms) / len(norms)
                self.tb_writer.add_scalar(f"gradients/{layer_name}", avg_norm, step)
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: int):
        """Log learning rate from optimizer."""
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group["lr"]
            self.log_metrics({f"lr_group_{i}": lr}, step=step)
    
    def save_history(self):
        """Save full training history to JSON."""
        history_file = self.logs_dir / "training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def close(self):
        """Close all logging backends."""
        self.save_history()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            import wandb
            wandb.finish()
        
        elapsed = time.time() - self._training_start_time
        self.logger.info(f"Training completed in {elapsed/3600:.2f} hours")


class ProgressTracker:
    """
    Track training progress with ETA estimation.
    """
    
    def __init__(self, total_steps: int, smoothing: float = 0.1):
        self.total_steps = total_steps
        self.smoothing = smoothing
        
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.ema_step_time: Optional[float] = None
        self._last_step_time = self.start_time
    
    def update(self, current_step: int) -> Dict[str, Any]:
        """
        Update progress and return stats.
        
        Returns:
            Dict with progress stats including ETA
        """
        now = time.time()
        step_time = now - self._last_step_time
        self._last_step_time = now
        
        # Update EMA
        if self.ema_step_time is None:
            self.ema_step_time = step_time
        else:
            self.ema_step_time = self.smoothing * step_time + (1 - self.smoothing) * self.ema_step_time
        
        # Calculate progress
        progress = current_step / self.total_steps
        elapsed = now - self.start_time
        
        # ETA
        remaining_steps = self.total_steps - current_step
        eta_seconds = remaining_steps * self.ema_step_time if self.ema_step_time else 0
        
        return {
            "progress": progress,
            "elapsed_hours": elapsed / 3600,
            "eta_hours": eta_seconds / 3600,
            "steps_per_second": 1 / self.ema_step_time if self.ema_step_time else 0,
        }


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu_memory_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dict for console output."""
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if "loss" in key or "lr" in key:
                parts.append(f"{key}={value:.4f}")
            elif "ppl" in key or "perplexity" in key:
                parts.append(f"{key}={value:.2f}")
            else:
                parts.append(f"{key}={value:.3g}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
