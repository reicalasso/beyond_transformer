"""
Advanced Visualization Tools for PULSE Components

This module provides specialized visualization tools for PULSE-specific components.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class AdvancedPULSEVisualizer:
    """
    Advanced visualizer for PULSE components.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "whitegrid"):
        """
        Initialize advanced visualizer.

        Args:
            figsize: Figure size for plots
            style: Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
        self.palette = sns.color_palette("Set2", 10)

    def plot_token_to_state_routing(
        self,
        routing_weights: torch.Tensor,
        token_labels: Optional[List[str]] = None,
        state_labels: Optional[List[str]] = None,
        title: str = "Token-to-State Routing",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot token-to-state routing weights.

        Args:
            routing_weights: Routing weights tensor [batch, seq_len, num_states] or [seq_len, num_states]
            token_labels: Labels for tokens
            state_labels: Labels for states
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if routing_weights.dim() == 3:
            # [batch, seq_len, num_states] - take first batch
            weights = routing_weights[0].detach().cpu().numpy()
        else:
            # [seq_len, num_states]
            weights = routing_weights.detach().cpu().numpy()

        seq_len, num_states = weights.shape

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot heatmap
        im = sns.heatmap(
            weights, annot=False, cmap="Blues", square=False, cbar=True, ax=ax
        )

        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("State Nodes", fontsize=12)
        ax.set_ylabel("Input Tokens", fontsize=12)

        # Set custom labels if provided
        if state_labels:
            ax.set_xticks(range(min(num_states, len(state_labels))))
            ax.set_xticklabels(state_labels[:num_states], rotation=45, ha="right")
        else:
            ax.set_xticks(range(num_states))
            ax.set_xticklabels(
                [f"State {i}" for i in range(num_states)], rotation=45, ha="right"
            )

        if token_labels:
            ax.set_yticks(range(min(seq_len, len(token_labels))))
            ax.set_yticklabels(token_labels[:seq_len], rotation=0)
        else:
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels([f"Token {i}" for i in range(seq_len)], rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig

    def plot_state_communication(
        self,
        attention_weights: torch.Tensor,
        state_labels: Optional[List[str]] = None,
        title: str = "State-to-State Communication",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot state-to-state communication patterns.

        Args:
            attention_weights: Attention weights tensor [batch, heads, states, states] or [states, states]
            state_labels: Labels for states
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if attention_weights.dim() == 4:
            # [batch, heads, states, states] - take first batch and head
            weights = attention_weights[0, 0].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            # [heads, states, states] - take first head
            weights = attention_weights[0].detach().cpu().numpy()
        else:
            # [states, states]
            weights = attention_weights.detach().cpu().numpy()

        num_states = weights.shape[0]

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot heatmap
        im = sns.heatmap(
            weights, annot=False, cmap="Reds", square=True, cbar=True, ax=ax
        )

        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Target States", fontsize=12)
        ax.set_ylabel("Source States", fontsize=12)

        # Set custom labels if provided
        if state_labels:
            ax.set_xticks(range(min(num_states, len(state_labels))))
            ax.set_xticklabels(state_labels[:num_states], rotation=45, ha="right")
            ax.set_yticks(range(min(num_states, len(state_labels))))
            ax.set_yticklabels(state_labels[:num_states], rotation=0)
        else:
            ax.set_xticks(range(num_states))
            ax.set_xticklabels(
                [f"State {i}" for i in range(num_states)], rotation=45, ha="right"
            )
            ax.set_yticks(range(num_states))
            ax.set_yticklabels([f"State {i}" for i in range(num_states)], rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig

    def plot_memory_read_write_operations(
        self,
        read_weights: torch.Tensor,
        write_weights: torch.Tensor,
        memory_slots: Optional[List[str]] = None,
        title: str = "Memory Read/Write Operations",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot memory read and write operations.

        Args:
            read_weights: Read weights tensor [heads, slots] or [slots]
            write_weights: Write weights tensor [heads, slots] or [slots]
            memory_slots: Labels for memory slots
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if read_weights.dim() == 2:
            read_w = read_weights[0].detach().cpu().numpy()
        else:
            read_w = read_weights.detach().cpu().numpy()

        if write_weights.dim() == 2:
            write_w = write_weights[0].detach().cpu().numpy()
        else:
            write_w = write_weights.detach().cpu().numpy()

        num_slots = max(len(read_w), len(write_w))

        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, num_slots * 0.4), 6))

        # Create bar plot
        x = np.arange(num_slots)
        width = 0.35

        ax.bar(
            x - width / 2,
            read_w[:num_slots],
            width,
            label="Read",
            color=self.palette[0],
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            write_w[:num_slots],
            width,
            label="Write",
            color=self.palette[1],
            alpha=0.8,
        )

        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Memory Slots", fontsize=12)
        ax.set_ylabel("Attention Weight", fontsize=12)
        ax.legend()

        # Set custom labels if provided
        if memory_slots:
            ax.set_xticks(range(min(num_slots, len(memory_slots))))
            ax.set_xticklabels(memory_slots[:num_slots], rotation=45, ha="right")
        else:
            ax.set_xticks(range(num_slots))
            ax.set_xticklabels(
                [f"Slot {i}" for i in range(num_slots)], rotation=45, ha="right"
            )

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig

    def plot_state_dynamics(
        self,
        state_trajectories: List[torch.Tensor],
        state_labels: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        title: str = "State Dynamics",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot state dynamics over time.

        Args:
            state_trajectories: List of state tensors [slots, dim] for each time step
            state_labels: Labels for states
            metrics: Metrics to plot ('norm', 'mean', 'std', 'max', 'min')
            title: Plot title
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        if not metrics:
            metrics = ["norm", "mean"]

        # Convert trajectories to numpy
        trajectories = [state.detach().cpu().numpy() for state in state_trajectories]
        num_time_steps = len(trajectories)
        num_states = trajectories[0].shape[0] if trajectories else 0

        # Calculate metrics for each time step
        metric_data = {metric: [] for metric in metrics}
        time_steps = range(num_time_steps)

        for trajectory in trajectories:
            for metric in metrics:
                if metric == "norm":
                    values = np.linalg.norm(trajectory, axis=1)
                elif metric == "mean":
                    values = np.mean(trajectory, axis=1)
                elif metric == "std":
                    values = np.std(trajectory, axis=1)
                elif metric == "max":
                    values = np.max(trajectory, axis=1)
                elif metric == "min":
                    values = np.min(trajectory, axis=1)
                else:
                    values = np.mean(trajectory, axis=1)
                metric_data[metric].append(values)

        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Plot for first few states
            num_plot_states = min(6, num_states)
            for state_idx in range(num_plot_states):
                values = [metric_data[metric][t][state_idx] for t in time_steps]
                ax.plot(
                    time_steps,
                    values,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=f"State {state_idx}",
                    color=self.palette[state_idx],
                )

            ax.set_title(f"{title} - {metric.capitalize()}", fontsize=12)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(metric.capitalize())
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return fig

    def create_comprehensive_report(
        self,
        visualization_data: Dict[str, Any],
        save_dir: str = "visualization_reports",
    ) -> str:
        """
        Create comprehensive visualization report.

        Args:
            visualization_data: Dictionary with all visualization data
            save_dir: Directory to save report files

        Returns:
            Path to report directory
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        # Create individual visualizations
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_dir = save_path / f"pulse_visualization_report_{timestamp}"
        report_dir.mkdir(exist_ok=True)

        print(f"Creating comprehensive visualization report in: {report_dir}")

        # Plot attention maps if available
        if "attention_weights" in visualization_data:
            try:
                self.plot_attention_map(
                    visualization_data["attention_weights"],
                    title="Attention Weights Heatmap",
                    save_path=report_dir / "attention_heatmap.png",
                )
                print("✅ Attention heatmap saved")
            except Exception as e:
                print(f"⚠️ Could not save attention heatmap: {e}")

        # Plot token-to-state routing if available
        if "routing_weights" in visualization_data:
            try:
                self.plot_token_to_state_routing(
                    visualization_data["routing_weights"],
                    title="Token-to-State Routing",
                    save_path=report_dir / "token_state_routing.png",
                )
                print("✅ Token-state routing saved")
            except Exception as e:
                print(f"⚠️ Could not save token-state routing: {e}")

        # Plot memory content if available
        if "memory_content" in visualization_data:
            try:
                self.plot_memory_content(
                    visualization_data["memory_content"],
                    title="Memory Content",
                    save_path=report_dir / "memory_content.png",
                )
                print("✅ Memory content saved")
            except Exception as e:
                print(f"⚠️ Could not save memory content: {e}")

        # Plot state communication if available
        if "state_attention" in visualization_data:
            try:
                self.plot_state_communication(
                    visualization_data["state_attention"],
                    title="State-to-State Communication",
                    save_path=report_dir / "state_communication.png",
                )
                print("✅ State communication saved")
            except Exception as e:
                print(f"⚠️ Could not save state communication: {e}")

        # Create summary statistics
        try:
            summary_df = self.create_summary_statistics(visualization_data)
            summary_df.to_csv(report_dir / "summary_statistics.csv", index=False)
            print("✅ Summary statistics saved")
        except Exception as e:
            print(f"⚠️ Could not save summary statistics: {e}")

        # Create README with visualization descriptions
        readme_content = f"""# PULSE Visualization Report

Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Available Visualizations

1. **Attention Heatmap** - Shows attention patterns between tokens
2. **Token-to-State Routing** - Displays how tokens are routed to states
3. **Memory Content** - Visualizes memory slot contents
4. **State Communication** - Shows state-to-state interaction patterns
5. **Summary Statistics** - Numerical overview of key metrics

## Interpretation Guide

- **Attention Heatmaps**: Brighter colors indicate stronger attention connections
- **Routing Patterns**: Higher values show stronger token-state associations
- **Memory Content**: Color intensity represents value magnitude (red=positive, blue=negative)
- **State Communication**: Shows how states influence each other during processing

This report provides insights into the internal workings of the PULSE.
"""

        with open(report_dir / "README.md", "w") as f:
            f.write(readme_content)

        print("✅ Report README created")
        print(f"\n📊 Visualization report saved to: {report_dir}")

        return str(report_dir)

    def create_summary_statistics(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Create summary statistics for visualization data.

        Args:
            data: Dictionary with visualization data

        Returns:
            Pandas DataFrame with summary statistics
        """
        summary_data = []

        # Process each data type
        for key, value in data.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                try:
                    np_value = value.detach().cpu().numpy()
                    summary_data.append(
                        {
                            "Component": key,
                            "Shape": str(np_value.shape),
                            "Mean": float(np.mean(np_value)),
                            "Std": float(np.std(np_value)),
                            "Min": float(np.min(np_value)),
                            "Max": float(np.max(np_value)),
                            "Elements": int(np_value.size),
                        }
                    )
                except Exception as e:
                    print(f"Warning: Could not process {key}: {e}")

        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    print("Testing Advanced PULSE Visualizer...")

    # Create visualizer
    visualizer = AdvancedPULSEVisualizer()

    # Test token-to-state routing
    routing_weights = torch.softmax(torch.randn(10, 8), dim=-1)  # 10 tokens, 8 states
    fig1 = visualizer.plot_token_to_state_routing(
        routing_weights,
        token_labels=[f"Token{i}" for i in range(10)],
        state_labels=[f"State{i}" for i in range(8)],
        title="Test Token-to-State Routing",
    )

    # Test state communication
    state_attention = torch.softmax(torch.randn(8, 8), dim=-1)  # 8 states
    fig2 = visualizer.plot_state_communication(
        state_attention,
        state_labels=[f"S{i}" for i in range(8)],
        title="Test State Communication",
    )

    # Test memory operations
    read_weights = torch.softmax(torch.randn(6), dim=0)  # 6 memory slots
    write_weights = torch.softmax(torch.randn(6), dim=0)
    fig3 = visualizer.plot_memory_read_write_operations(
        read_weights,
        write_weights,
        memory_slots=[f"M{i}" for i in range(6)],
        title="Test Memory Operations",
    )

    # Test state dynamics
    state_trajectories = [
        torch.randn(5, 10) for _ in range(8)
    ]  # 5 states, 10 dims, 8 time steps
    fig4 = visualizer.plot_state_dynamics(
        state_trajectories,
        state_labels=[f"State{i}" for i in range(5)],
        metrics=["norm", "mean", "std"],
        title="Test State Dynamics",
    )

    # Test comprehensive report
    test_data = {
        "attention_weights": torch.softmax(torch.randn(8, 8), dim=-1),
        "routing_weights": torch.softmax(torch.randn(10, 8), dim=-1),
        "memory_content": torch.randn(16, 20),
        "state_attention": torch.softmax(torch.randn(8, 8), dim=-1),
    }

    report_dir = visualizer.create_comprehensive_report(test_data, "test_reports")
    print(f"Report created in: {report_dir}")

    print("✅ Advanced PULSE Visualizer test completed!")
