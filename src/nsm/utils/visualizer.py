"""
Visualization Tools for Neural State Machine Interpretability

This module provides tools for visualizing attention maps and memory contents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path


class NSMVisualizer:
    """
    Visualizer for Neural State Machine interpretability.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "whitegrid"):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
            style: Seaborn style
        """
        self.figsize = figsize
        sns.set_style(style)
        
        # Color palette for consistent visualization
        self.palette = sns.color_palette("husl", 10)
    
    def plot_attention_map(self, attention_weights: torch.Tensor, 
                          title: str = "Attention Map",
                          x_labels: Optional[List[str]] = None,
                          y_labels: Optional[List[str]] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot attention map as heatmap.
        
        Args:
            attention_weights: Attention weights tensor [batch, heads, seq, seq] or [seq, seq]
            title: Plot title
            x_labels: Labels for x-axis
            y_labels: Labels for y-axis
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if attention_weights.dim() == 4:
            # [batch, heads, seq, seq] - take first batch and head
            attention_map = attention_weights[0, 0].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            # [heads, seq, seq] - take first head
            attention_map = attention_weights[0].detach().cpu().numpy()
        else:
            # [seq, seq]
            attention_map = attention_weights.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        im = sns.heatmap(
            attention_map,
            annot=False,
            cmap="viridis",
            square=True,
            cbar=True,
            ax=ax
        )
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Key Positions", fontsize=12)
        ax.set_ylabel("Query Positions", fontsize=12)
        
        # Set custom labels if provided
        if x_labels:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")
        if y_labels:
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_memory_content(self, memory: torch.Tensor,
                          title: str = "Memory Content",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot memory content as heatmap.
        
        Args:
            memory: Memory tensor [slots, dim] or [batch, slots, dim]
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Handle different tensor shapes
        if memory.dim() == 3:
            # [batch, slots, dim] - take first batch
            memory_content = memory[0].detach().cpu().numpy()
        else:
            # [slots, dim]
            memory_content = memory.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot heatmap
        im = sns.heatmap(
            memory_content,
            annot=False,
            cmap="RdBu_r",
            center=0,
            cbar=True,
            ax=ax
        )
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Memory Dimensions", fontsize=12)
        ax.set_ylabel("Memory Slots", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_state_evolution(self, states: List[torch.Tensor],
                           title: str = "State Evolution",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot state evolution over time/process steps.
        
        Args:
            states: List of state tensors [slots, dim]
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy arrays
        state_arrays = [state.detach().cpu().numpy() for state in states]
        
        # Calculate statistics
        mean_states = [np.mean(state, axis=1) for state in state_arrays]  # Mean per slot
        time_steps = range(len(mean_states))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot evolution for first few slots
        num_slots = min(8, mean_states[0].shape[0])
        for slot_idx in range(num_slots):
            slot_evolution = [mean_states[t][slot_idx] for t in time_steps]
            ax.plot(time_steps, slot_evolution, 
                   marker='o', linewidth=2, markersize=6,
                   label=f'Slot {slot_idx}', color=self.palette[slot_idx])
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Processing Steps", fontsize=12)
        ax.set_ylabel("Average State Value", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_attention_patterns(self, attention_weights: torch.Tensor,
                              pattern_names: Optional[List[str]] = None,
                              title: str = "Attention Patterns",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot multiple attention patterns.
        
        Args:
            attention_weights: Attention weights tensor [patterns, seq, seq]
            pattern_names: Names for each pattern
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        attention_arrays = attention_weights.detach().cpu().numpy()
        num_patterns = attention_arrays.shape[0]
        
        # Determine subplot layout
        cols = min(3, num_patterns)
        rows = (num_patterns + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if num_patterns == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Plot each pattern
        for i in range(num_patterns):
            if i < len(axes):
                ax = axes[i]
                im = sns.heatmap(
                    attention_arrays[i],
                    annot=False,
                    cmap="viridis",
                    square=True,
                    cbar=True,
                    ax=ax
                )
                pattern_name = pattern_names[i] if pattern_names and i < len(pattern_names) else f"Pattern {i+1}"
                ax.set_title(pattern_name, fontsize=12)
                ax.set_xlabel("Key Positions")
                ax.set_ylabel("Query Positions")
        
        # Hide unused subplots
        for i in range(num_patterns, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def plot_memory_importance(self, importance_scores: torch.Tensor,
                             title: str = "Memory Slot Importance",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot memory slot importance scores.
        
        Args:
            importance_scores: Importance scores tensor [slots]
            title: Plot title
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy
        scores = importance_scores.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, len(scores) * 0.3), 6))
        
        # Create bar plot
        bars = ax.bar(range(len(scores)), scores, color=self.palette[0])
        
        # Set labels
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Memory Slots", fontsize=12)
        ax.set_ylabel("Importance Score", fontsize=12)
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels([f"Slot {i}" for i in range(len(scores))])
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def create_interactive_summary(self, data: Dict[str, Any],
                                 save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create interactive summary table.
        
        Args:
            data: Dictionary with visualization data
            save_path: Path to save CSV (optional)
            
        Returns:
            Pandas DataFrame with summary
        """
        # Create summary statistics
        summary_data = []
        
        if 'attention_weights' in data:
            attn = data['attention_weights']
            if attn.dim() >= 2:
                attn_np = attn.detach().cpu().numpy()
                summary_data.append({
                    'Metric': 'Attention Weights',
                    'Mean': np.mean(attn_np),
                    'Std': np.std(attn_np),
                    'Min': np.min(attn_np),
                    'Max': np.max(attn_np),
                    'Shape': str(attn_np.shape)
                })
        
        if 'memory_content' in data:
            mem = data['memory_content']
            if mem.dim() >= 2:
                mem_np = mem.detach().cpu().numpy()
                summary_data.append({
                    'Metric': 'Memory Content',
                    'Mean': np.mean(mem_np),
                    'Std': np.std(mem_np),
                    'Min': np.min(mem_np),
                    'Max': np.max(mem_np),
                    'Shape': str(mem_np.shape)
                })
        
        if 'importance_scores' in data:
            imp = data['importance_scores']
            if imp.dim() >= 1:
                imp_np = imp.detach().cpu().numpy()
                summary_data.append({
                    'Metric': 'Importance Scores',
                    'Mean': np.mean(imp_np),
                    'Std': np.std(imp_np),
                    'Min': np.min(imp_np),
                    'Max': np.max(imp_np),
                    'Shape': str(imp_np.shape)
                })
        
        df = pd.DataFrame(summary_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
        
        return df


# Example usage
if __name__ == "__main__":
    print("Testing NSM Visualizer...")
    
    # Create visualizer
    visualizer = NSMVisualizer()
    
    # Test attention map
    attention_weights = torch.softmax(torch.randn(4, 4), dim=-1)
    fig1 = visualizer.plot_attention_map(attention_weights, "Test Attention Map")
    
    # Test memory content
    memory_content = torch.randn(8, 16)
    fig2 = visualizer.plot_memory_content(memory_content, "Test Memory Content")
    
    # Test state evolution
    states = [torch.randn(8, 16) for _ in range(5)]
    fig3 = visualizer.plot_state_evolution(states, "Test State Evolution")
    
    # Test attention patterns
    attention_patterns = torch.softmax(torch.randn(3, 6, 6), dim=-1)
    fig4 = visualizer.plot_attention_patterns(attention_patterns, 
                                            ["Pattern 1", "Pattern 2", "Pattern 3"],
                                            "Test Attention Patterns")
    
    # Test memory importance
    importance_scores = torch.sigmoid(torch.randn(8))
    fig5 = visualizer.plot_memory_importance(importance_scores, "Test Memory Importance")
    
    # Test summary
    test_data = {
        'attention_weights': attention_weights,
        'memory_content': memory_content,
        'importance_scores': importance_scores
    }
    summary_df = visualizer.create_interactive_summary(test_data)
    print("Summary DataFrame:")
    print(summary_df)
    
    print("✅ NSM Visualizer test completed!")