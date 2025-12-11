"""
Attention Visualization for PULSEs.

This module provides tools for visualizing attention patterns:
- Token-to-state routing visualization
- Self-attention patterns
- Cross-attention between tokens and states
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class AttentionVisualizer:
    """
    Visualizer for PULSE attention patterns.
    
    Provides methods to visualize:
    - Self-attention weights
    - Token-to-state routing weights
    - Cross-attention patterns
    - Attention head analysis
    """

    def __init__(
        self,
        model: nn.Module,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = "Blues",
    ) -> None:
        """
        Initialize AttentionVisualizer.

        Args:
            model: The PULSE model to visualize.
            figsize: Default figure size.
            cmap: Default colormap.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

        self.model = model
        self.figsize = figsize
        self.cmap = cmap

        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")

        self.captured_attention: List[torch.Tensor] = []
        self.hooks: List[Any] = []

    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        self.captured_attention = []
        self.hooks = []

        def capture_hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                # Check if second element looks like attention weights
                attn = output[1] if len(output) > 1 else None
                if attn is not None and isinstance(attn, torch.Tensor):
                    if attn.dim() >= 3:  # [batch, heads, seq, seq] or similar
                        self.captured_attention.append(attn.detach().cpu())

        for name, module in self.model.named_modules():
            if "attention" in name.lower():
                hook = module.register_forward_hook(capture_hook)
                self.hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def capture_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Capture attention weights during forward pass.

        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask

        Returns:
            List of attention weight tensors
        """
        self._register_attention_hooks()

        self.model.eval()
        with torch.no_grad():
            if attention_mask is not None:
                self.model(input_ids, attention_mask=attention_mask)
            else:
                self.model(input_ids)

        self._remove_hooks()

        return self.captured_attention

    def plot_attention_heatmap(
        self,
        attention: torch.Tensor,
        tokens: Optional[List[str]] = None,
        head_idx: int = 0,
        layer_idx: int = 0,
        title: str = "Attention Weights",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot attention weights as a heatmap.

        Args:
            attention: Attention tensor [batch, heads, seq, seq] or [heads, seq, seq]
            tokens: Optional token labels
            head_idx: Which attention head to visualize
            layer_idx: Layer index (for title)
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if attention.dim() == 4:
            attn = attention[0, head_idx].numpy()
        elif attention.dim() == 3:
            attn = attention[head_idx].numpy()
        else:
            attn = attention.numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        if SEABORN_AVAILABLE:
            sns.heatmap(
                attn,
                ax=ax,
                cmap=self.cmap,
                xticklabels=tokens if tokens else False,
                yticklabels=tokens if tokens else False,
                square=True,
            )
        else:
            im = ax.imshow(attn, cmap=self.cmap)
            plt.colorbar(im, ax=ax)
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right")
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)

        ax.set_title(f"{title} (Layer {layer_idx}, Head {head_idx})")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_heads(
        self,
        attention: torch.Tensor,
        tokens: Optional[List[str]] = None,
        max_heads: int = 8,
        title: str = "Attention Heads",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot multiple attention heads in a grid.

        Args:
            attention: Attention tensor [batch, heads, seq, seq]
            tokens: Optional token labels
            max_heads: Maximum number of heads to show
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if attention.dim() == 4:
            attention = attention[0]

        num_heads = min(attention.shape[0], max_heads)
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        axes = np.array(axes).flatten() if num_heads > 1 else [axes]

        for i in range(num_heads):
            attn = attention[i].numpy()
            ax = axes[i]

            im = ax.imshow(attn, cmap=self.cmap)
            ax.set_title(f"Head {i}")
            ax.set_xlabel("Key")
            ax.set_ylabel("Query")

        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis("off")

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_routing_weights(
        self,
        routing_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        title: str = "Token-to-State Routing",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot token-to-state routing weights.

        Args:
            routing_weights: Routing tensor [batch, seq_len, num_states]
            tokens: Optional token labels
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if routing_weights.dim() == 3:
            weights = routing_weights[0].numpy()
        else:
            weights = routing_weights.numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        if SEABORN_AVAILABLE:
            sns.heatmap(
                weights,
                ax=ax,
                cmap=self.cmap,
                yticklabels=tokens if tokens else False,
            )
        else:
            im = ax.imshow(weights, aspect="auto", cmap=self.cmap)
            plt.colorbar(im, ax=ax, label="Routing Weight")
            if tokens:
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens)

        ax.set_xlabel("State Index")
        ax.set_ylabel("Token Position")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_entropy(
        self,
        attention: torch.Tensor,
        title: str = "Attention Entropy",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot attention entropy for each head and position.

        Args:
            attention: Attention tensor [batch, heads, seq, seq]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if attention.dim() == 4:
            attention = attention[0]

        # Compute entropy: -sum(p * log(p))
        eps = 1e-8
        entropy = -(attention * torch.log(attention + eps)).sum(dim=-1)
        entropy = entropy.numpy()

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(entropy, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Query Position")
        ax.set_ylabel("Attention Head")
        ax.set_title(title)

        plt.colorbar(im, ax=ax, label="Entropy (bits)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_flow(
        self,
        attention_list: List[torch.Tensor],
        position: int = 0,
        title: str = "Attention Flow Across Layers",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot how attention flows across layers for a specific position.

        Args:
            attention_list: List of attention tensors from each layer
            position: Query position to track
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        num_layers = len(attention_list)

        # Extract attention for the specific position from each layer
        attention_flow = []
        for attn in attention_list:
            if attn.dim() == 4:
                attn = attn[0]
            # Average across heads
            avg_attn = attn.mean(dim=0)[position].numpy()
            attention_flow.append(avg_attn)

        attention_flow = np.array(attention_flow)

        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(attention_flow, aspect="auto", cmap=self.cmap)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title} (Query Position {position})")

        plt.colorbar(im, ax=ax, label="Attention Weight")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_head_importance(
        self,
        attention: torch.Tensor,
        title: str = "Attention Head Importance",
        save_path: Optional[str] = None,
    ) -> Figure:
        """
        Plot importance of each attention head based on attention variance.

        Args:
            attention: Attention tensor [batch, heads, seq, seq]
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure
        """
        if attention.dim() == 4:
            attention = attention[0]

        # Compute importance as variance of attention patterns
        importance = attention.var(dim=(1, 2)).numpy()
        num_heads = len(importance)

        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.get_cmap(self.cmap)(importance / importance.max())
        bars = ax.bar(range(num_heads), importance, color=colors)

        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Importance (Variance)")
        ax.set_title(title)

        # Add mean line
        mean_importance = importance.mean()
        ax.axhline(y=mean_importance, color="red", linestyle="--", 
                   label=f"Mean: {mean_importance:.4f}")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def create_attention_report(
        self,
        input_ids: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_dir: str = "./attention_report",
    ) -> None:
        """
        Create a comprehensive attention visualization report.

        Args:
            input_ids: Input token IDs
            tokens: Optional token labels
            save_dir: Directory to save report
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Capture attention
        attention_list = self.capture_attention(input_ids)

        if not attention_list:
            print("No attention weights captured.")
            return

        # Generate visualizations
        for i, attn in enumerate(attention_list):
            # Heatmap for first head
            self.plot_attention_heatmap(
                attn,
                tokens=tokens,
                head_idx=0,
                layer_idx=i,
                save_path=os.path.join(save_dir, f"layer_{i}_head_0.png"),
            )
            plt.close()

            # All heads
            self.plot_attention_heads(
                attn,
                tokens=tokens,
                title=f"Layer {i} Attention Heads",
                save_path=os.path.join(save_dir, f"layer_{i}_all_heads.png"),
            )
            plt.close()

            # Entropy
            self.plot_attention_entropy(
                attn,
                title=f"Layer {i} Attention Entropy",
                save_path=os.path.join(save_dir, f"layer_{i}_entropy.png"),
            )
            plt.close()

        # Attention flow
        self.plot_attention_flow(
            attention_list,
            position=0,
            save_path=os.path.join(save_dir, "attention_flow.png"),
        )
        plt.close()

        print(f"Attention report saved to {save_dir}")
