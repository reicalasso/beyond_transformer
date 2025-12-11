"""
State Visualization for PULSEs.

This module provides tools for visualizing PULSE state dynamics:
- State activation heatmaps
- State evolution over time
- State importance analysis
- State clustering and similarity
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class StateVisualizer:
    """
    Visualizer for PULSE state dynamics.
    
    Provides methods to visualize:
    - State activations at each layer
    - State evolution over sequence positions
    - State importance scores
    - State similarity matrices
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "viridis",
        style: str = "whitegrid",
    ) -> None:
        """
        Initialize StateVisualizer.

        Args:
            model: The PULSE model to visualize.
            figsize: Default figure size.
            cmap: Default colormap.
            style: Seaborn style.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

        self.model = model
        self.figsize = figsize
        self.cmap = cmap
        self.style = style

        if SEABORN_AVAILABLE:
            sns.set_style(style)

        # Storage for captured states
        self.captured_states: List[torch.Tensor] = []
        self.hooks: List[Any] = []

    def _register_hooks(self) -> None:
        """Register forward hooks to capture states."""
        self.captured_states = []
        self.hooks = []

        def capture_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                # Assume second element is states
                states = output[1]
                if isinstance(states, torch.Tensor):
                    self.captured_states.append(states.detach().cpu())

        # Register hooks on PULSE layers
        for name, module in self.model.named_modules():
            if "pulse" in name.lower() or "state" in name.lower():
                hook = module.register_forward_hook(capture_hook)
                self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def capture_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Capture states during a forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask

        Returns:
            List of state tensors from each layer
        """
        self._register_hooks()

        self.model.eval()
        with torch.no_grad():
            if attention_mask is not None:
                self.model(input_ids, attention_mask=attention_mask, output_states=True)
            else:
                self.model(input_ids, output_states=True)

        self._remove_hooks()

        return self.captured_states

    def plot_state_activations(
        self,
        states: torch.Tensor,
        title: str = "State Activations",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot state activation heatmap.

        Args:
            states: State tensor [batch_size, num_states, state_dim] or [num_states, state_dim]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if states.dim() == 3:
            states = states[0]  # Take first batch item

        states_np = states.numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(states_np, aspect="auto", cmap=self.cmap)
        ax.set_xlabel("State Dimension")
        ax.set_ylabel("State Index")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Activation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_state_evolution(
        self,
        states_list: List[torch.Tensor],
        state_idx: int = 0,
        title: str = "State Evolution Across Layers",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot how a specific state evolves across layers.

        Args:
            states_list: List of state tensors from each layer
            state_idx: Index of state to track
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        num_layers = len(states_list)

        # Extract the specific state from each layer
        state_evolution = []
        for states in states_list:
            if states.dim() == 3:
                states = states[0]  # First batch item
            state_evolution.append(states[state_idx].numpy())

        state_evolution = np.array(state_evolution)

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(state_evolution, aspect="auto", cmap=self.cmap)
        ax.set_xlabel("State Dimension")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title} (State {state_idx})")

        plt.colorbar(im, ax=ax, label="Activation")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_state_importance(
        self,
        importance_scores: torch.Tensor,
        title: str = "State Importance Scores",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot state importance scores as a bar chart.

        Args:
            importance_scores: Importance scores [num_states]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        scores = importance_scores.numpy()
        num_states = len(scores)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.get_cmap(self.cmap)(scores / scores.max())
        bars = ax.bar(range(num_states), scores, color=colors)

        ax.set_xlabel("State Index")
        ax.set_ylabel("Importance Score")
        ax.set_title(title)

        # Add threshold line
        threshold = scores.mean()
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"Mean: {threshold:.3f}")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_state_similarity(
        self,
        states: torch.Tensor,
        title: str = "State Similarity Matrix",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot state similarity matrix.

        Args:
            states: State tensor [num_states, state_dim]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if states.dim() == 3:
            states = states[0]

        # Compute cosine similarity
        states_norm = states / states.norm(dim=1, keepdim=True).clamp(min=1e-8)
        similarity = torch.mm(states_norm, states_norm.t()).numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        if SEABORN_AVAILABLE:
            sns.heatmap(
                similarity,
                ax=ax,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                annot=False,
            )
        else:
            im = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, label="Cosine Similarity")

        ax.set_xlabel("State Index")
        ax.set_ylabel("State Index")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_state_pca(
        self,
        states: torch.Tensor,
        labels: Optional[List[str]] = None,
        title: str = "State PCA Projection",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot PCA projection of states.

        Args:
            states: State tensor [num_states, state_dim]
            labels: Optional labels for each state
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if states.dim() == 3:
            states = states[0]

        states_np = states.numpy()

        # Simple PCA using SVD
        states_centered = states_np - states_np.mean(axis=0)
        U, S, Vt = np.linalg.svd(states_centered, full_matrices=False)
        pca_coords = U[:, :2] * S[:2]

        fig, ax = plt.subplots(figsize=self.figsize)

        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=range(len(pca_coords)),
            cmap=self.cmap,
            s=100,
            alpha=0.7,
        )

        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (pca_coords[i, 0], pca_coords[i, 1]))
        else:
            for i in range(len(pca_coords)):
                ax.annotate(str(i), (pca_coords[i, 0], pca_coords[i, 1]))

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)

        plt.colorbar(scatter, ax=ax, label="State Index")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_state_dynamics(
        self,
        input_ids: torch.Tensor,
        title: str = "State Dynamics",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Comprehensive visualization of state dynamics for an input.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Capture states
        states_list = self.capture_states(input_ids)

        if not states_list:
            raise ValueError("No states captured. Make sure model has state outputs.")

        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5))

        # Plot 1: First layer state activations
        self._plot_on_axis(
            axes[0, 0],
            states_list[0][0].numpy() if states_list[0].dim() == 3 else states_list[0].numpy(),
            "Layer 1 State Activations",
        )

        # Plot 2: Last layer state activations
        self._plot_on_axis(
            axes[0, 1],
            states_list[-1][0].numpy() if states_list[-1].dim() == 3 else states_list[-1].numpy(),
            "Last Layer State Activations",
        )

        # Plot 3: State evolution for first state
        state_evolution = []
        for states in states_list:
            if states.dim() == 3:
                states = states[0]
            state_evolution.append(states[0].numpy())
        state_evolution = np.array(state_evolution)

        self._plot_on_axis(axes[1, 0], state_evolution, "State 0 Evolution Across Layers")

        # Plot 4: Final state similarity
        final_states = states_list[-1][0] if states_list[-1].dim() == 3 else states_list[-1]
        states_norm = final_states / final_states.norm(dim=1, keepdim=True).clamp(min=1e-8)
        similarity = torch.mm(states_norm, states_norm.t()).numpy()

        im = axes[1, 1].imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[1, 1].set_title("Final State Similarity")
        axes[1, 1].set_xlabel("State Index")
        axes[1, 1].set_ylabel("State Index")
        plt.colorbar(im, ax=axes[1, 1])

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def _plot_on_axis(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        title: str,
    ) -> None:
        """Helper to plot heatmap on a specific axis."""
        im = ax.imshow(data, aspect="auto", cmap=self.cmap)
        ax.set_title(title)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("Index")
        plt.colorbar(im, ax=ax)

    def create_animation(
        self,
        states_list: List[torch.Tensor],
        save_path: str,
        fps: int = 2,
    ) -> None:
        """
        Create an animation of state evolution.

        Args:
            states_list: List of state tensors from each layer
            save_path: Path to save animation (gif or mp4)
            fps: Frames per second
        """
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError("Animation requires matplotlib with animation support")

        fig, ax = plt.subplots(figsize=self.figsize)

        # Initialize with first frame
        states = states_list[0]
        if states.dim() == 3:
            states = states[0]
        im = ax.imshow(states.numpy(), aspect="auto", cmap=self.cmap)
        ax.set_xlabel("State Dimension")
        ax.set_ylabel("State Index")
        title = ax.set_title("Layer 0")
        plt.colorbar(im, ax=ax)

        def update(frame):
            states = states_list[frame]
            if states.dim() == 3:
                states = states[0]
            im.set_array(states.numpy())
            title.set_text(f"Layer {frame}")
            return [im, title]

        anim = FuncAnimation(
            fig,
            update,
            frames=len(states_list),
            interval=1000 // fps,
            blit=True,
        )

        if save_path.endswith(".gif"):
            anim.save(save_path, writer=PillowWriter(fps=fps))
        else:
            anim.save(save_path, fps=fps)

        plt.close(fig)
