"""
Logger for Neural State Machine Testing

This module provides logging functionality for tracking memory content,
attention weights, and state variables during testing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class NSMLogger:
    """
    Logger for Neural State Machine components.
    """

    def __init__(self, log_dir: str = "logs", experiment_name: str = "nsm_test"):
        """
        Initialize the NSMLogger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Log files
        self.metrics_log = self.experiment_dir / "metrics.json"
        self.memory_log = self.experiment_dir / "memory_log.json"
        self.attention_log = self.experiment_dir / "attention_log.json"
        self.states_log = self.experiment_dir / "states_log.json"

        # Initialize log files
        self._initialize_logs()

        # Storage for current batch data
        self.current_batch_data = {}

    def _initialize_logs(self):
        """Initialize log files."""
        # Create empty log files if they don't exist
        for log_file in [
            self.metrics_log,
            self.memory_log,
            self.attention_log,
            self.states_log,
        ]:
            if not log_file.exists():
                with open(log_file, "w") as f:
                    json.dump([], f)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log training metrics.

        Args:
            metrics: Dictionary of metrics
            step: Training step
        """
        log_entry = {"step": step, "metrics": metrics}

        self._append_to_log(self.metrics_log, log_entry)

    def log_memory_content(
        self, memory_content: torch.Tensor, step: int = None, batch_idx: int = None
    ):
        """
        Log memory content.

        Args:
            memory_content: Memory content tensor
            step: Training step
            batch_idx: Batch index
        """
        # Convert to numpy for serialization
        memory_np = memory_content.detach().cpu().numpy()

        log_entry = {
            "step": step,
            "batch_idx": batch_idx,
            "memory_shape": list(memory_np.shape),
            "memory_stats": {
                "mean": float(np.mean(memory_np)),
                "std": float(np.std(memory_np)),
                "min": float(np.min(memory_np)),
                "max": float(np.max(memory_np)),
            },
            "memory_sample": memory_np.flatten()[:100].tolist(),  # First 100 elements
        }

        self._append_to_log(self.memory_log, log_entry)

    def log_attention_weights(
        self,
        attention_weights: torch.Tensor,
        step: int = None,
        batch_idx: int = None,
        layer_name: str = "attention",
    ):
        """
        Log attention weights.

        Args:
            attention_weights: Attention weights tensor
            step: Training step
            batch_idx: Batch index
            layer_name: Name of the attention layer
        """
        # Convert to numpy for serialization
        attn_np = attention_weights.detach().cpu().numpy()

        log_entry = {
            "step": step,
            "batch_idx": batch_idx,
            "layer_name": layer_name,
            "attention_shape": list(attn_np.shape),
            "attention_stats": {
                "mean": float(np.mean(attn_np)),
                "std": float(np.std(attn_np)),
                "min": float(np.min(attn_np)),
                "max": float(np.max(attn_np)),
            },
            "attention_sample": attn_np.flatten()[:100].tolist(),  # First 100 elements
        }

        self._append_to_log(self.attention_log, log_entry)

    def log_state_variables(
        self,
        states: torch.Tensor,
        step: int = None,
        batch_idx: int = None,
        state_type: str = "hidden",
    ):
        """
        Log state variables.

        Args:
            states: State variables tensor
            step: Training step
            batch_idx: Batch index
            state_type: Type of states (e.g., "hidden", "cell", "memory")
        """
        # Convert to numpy for serialization
        states_np = states.detach().cpu().numpy()

        log_entry = {
            "step": step,
            "batch_idx": batch_idx,
            "state_type": state_type,
            "states_shape": list(states_np.shape),
            "states_stats": {
                "mean": float(np.mean(states_np)),
                "std": float(np.std(states_np)),
                "min": float(np.min(states_np)),
                "max": float(np.max(states_np)),
            },
            "states_sample": states_np.flatten()[:100].tolist(),  # First 100 elements
        }

        self._append_to_log(self.states_log, log_entry)

    def _append_to_log(self, log_file: Path, entry: Dict[str, Any]):
        """
        Append entry to log file.

        Args:
            log_file: Path to log file
            entry: Entry to append
        """
        # Read existing data
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        # Append new entry
        data.append(entry)

        # Write back to file
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2)

    def log_batch_data(self, batch_idx: int, data: Dict[str, Any]):
        """
        Log data for current batch.

        Args:
            batch_idx: Batch index
            data: Data to log
        """
        self.current_batch_data[batch_idx] = data

    def save_batch_visualizations(self, batch_idx: int = None):
        """
        Save visualizations for batch data.

        Args:
            batch_idx: Specific batch index to visualize, or None for all
        """
        vis_dir = self.experiment_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        batches_to_visualize = (
            [batch_idx]
            if batch_idx is not None
            else list(self.current_batch_data.keys())
        )

        for batch_id in batches_to_visualize:
            if batch_id in self.current_batch_data:
                self._create_batch_visualizations(batch_id, vis_dir)

    def _create_batch_visualizations(self, batch_idx: int, vis_dir: Path):
        """
        Create visualizations for a specific batch.

        Args:
            batch_idx: Batch index
            vis_dir: Directory to save visualizations
        """
        batch_data = self.current_batch_data[batch_idx]

        # Create attention heatmap if available
        if "attention_weights" in batch_data:
            attn_weights = batch_data["attention_weights"]
            if isinstance(attn_weights, torch.Tensor):
                attn_np = attn_weights.detach().cpu().numpy()

                # Handle multi-head attention
                if len(attn_np.shape) == 4:  # [batch, heads, seq, seq]
                    fig, axes = plt.subplots(
                        1, min(4, attn_np.shape[1]), figsize=(15, 4)
                    )
                    if attn_np.shape[1] == 1:
                        axes = [axes]

                    for i in range(min(4, attn_np.shape[1])):
                        sns.heatmap(attn_np[0, i], ax=axes[i], cmap="viridis")
                        axes[i].set_title(f"Attention Head {i}")
                        axes[i].set_xlabel("Key Position")
                        axes[i].set_ylabel("Query Position")

                    plt.tight_layout()
                    plt.savefig(vis_dir / f"attention_batch_{batch_idx}.png", dpi=150)
                    plt.close()

        # Create memory content visualization if available
        if "memory_content" in batch_data:
            memory = batch_data["memory_content"]
            if isinstance(memory, torch.Tensor):
                memory_np = memory.detach().cpu().numpy()

                if len(memory_np.shape) == 2:  # [slots, dim]
                    plt.figure(figsize=(12, 6))
                    sns.heatmap(memory_np, cmap="coolwarm", center=0)
                    plt.title("Memory Content")
                    plt.xlabel("Memory Dimension")
                    plt.ylabel("Memory Slot")
                    plt.savefig(vis_dir / f"memory_batch_{batch_idx}.png", dpi=150)
                    plt.close()

    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary of logged data.

        Returns:
            Dictionary with log summary
        """
        summary = {}

        # Metrics summary
        if self.metrics_log.exists():
            with open(self.metrics_log, "r") as f:
                metrics_data = json.load(f)
            summary["metrics_entries"] = len(metrics_data)
            if metrics_data:
                summary["latest_metrics"] = metrics_data[-1]["metrics"]

        # Memory summary
        if self.memory_log.exists():
            with open(self.memory_log, "r") as f:
                memory_data = json.load(f)
            summary["memory_entries"] = len(memory_data)

        # Attention summary
        if self.attention_log.exists():
            with open(self.attention_log, "r") as f:
                attention_data = json.load(f)
            summary["attention_entries"] = len(attention_data)

        # States summary
        if self.states_log.exists():
            with open(self.states_log, "r") as f:
                states_data = json.load(f)
            summary["states_entries"] = len(states_data)

        return summary


# Example usage
if __name__ == "__main__":
    print("Testing NSMLogger...")

    # Create logger
    logger = NSMLogger(experiment_name="test_experiment")

    # Test logging
    batch_size, seq_len, hidden_dim = 2, 5, 10

    # Log metrics
    logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)

    # Log memory content
    memory = torch.randn(32, 20)  # 32 slots, 20 dimensions
    logger.log_memory_content(memory, step=1, batch_idx=0)

    # Log attention weights
    attention = torch.softmax(
        torch.randn(batch_size, 4, seq_len, seq_len), dim=-1
    )  # 4 heads
    logger.log_attention_weights(
        attention, step=1, batch_idx=0, layer_name="self_attention"
    )

    # Log state variables
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    logger.log_state_variables(hidden_states, step=1, batch_idx=0, state_type="hidden")

    # Log batch data for visualization
    logger.log_batch_data(0, {"attention_weights": attention, "memory_content": memory})

    # Save visualizations
    logger.save_batch_visualizations(0)

    # Get summary
    summary = logger.get_log_summary()
    print(f"Log summary: {summary}")

    print("✅ NSMLogger test completed successfully!")
