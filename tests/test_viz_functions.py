#!/usr/bin/env python3
"""
Test script for visualization functions
"""

import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, "src")

from nsm import StateManager, TokenToStateRouter


def test_visualization_functions():
    """Test the visualization functions"""
    print("Testing visualization functions...")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create components
    router = TokenToStateRouter(token_dim=128, state_dim=64, num_states=16, num_heads=4)

    state_manager = StateManager(
        state_dim=64, max_states=16, initial_states=16, prune_threshold=0.3
    )

    # Create sample data
    batch_size = 4
    seq_len = 20
    sample_tokens = torch.randn(batch_size, seq_len, 128)
    sample_states = torch.randn(batch_size, 16, 64)

    # Test routing heatmap function
    def visualize_routing_heatmap(tokens, states, router, example_idx=0, title="Test"):
        with torch.no_grad():
            _, routing_weights = router(tokens, states)
            example_weights = routing_weights[example_idx].cpu().numpy()
            return example_weights

    print("Testing routing heatmap function...")
    weights = visualize_routing_heatmap(
        sample_tokens, sample_states, router, example_idx=0
    )
    print(f"   Routing weights shape: {weights.shape}")
    print(f"   Weight values range: {weights.min():.3f} - {weights.max():.3f}")

    # Test entropy function
    def compute_entropy(probs):
        eps = 1e-8
        probs = np.clip(probs, eps, 1.0)
        return -np.sum(probs * np.log(probs), axis=-1)

    print("Testing entropy computation...")
    entropies = compute_entropy(weights)
    print(f"   Entropies shape: {entropies.shape}")
    print(f"   Entropy values range: {entropies.min():.3f} - {entropies.max():.3f}")

    print("\nVisualization functions test passed!")


if __name__ == "__main__":
    test_visualization_functions()
