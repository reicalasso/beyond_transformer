#!/usr/bin/env python3
"""
Example: Visualizing PULSE State Dynamics

This script demonstrates how to visualize state activations and attention patterns.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from pulse.models.pulse_lm import PulseConfig, PulseForCausalLM
from pulse.visualization import StateVisualizer, AttentionVisualizer


def main():
    # Create a small model for visualization
    config = PulseConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_states=8,
        state_dim=128,
        intermediate_size=256,
        max_position_embeddings=128,
    )

    model = PulseForCausalLM(config)
    model.eval()

    # Create sample input
    text = "Hello world"
    input_ids = torch.tensor([[ord(c) % 1000 for c in text]])

    print("Creating visualizations...")
    os.makedirs("./visualizations", exist_ok=True)

    # State visualization
    print("1. Visualizing states...")
    state_viz = StateVisualizer(model, figsize=(10, 6))

    # Capture and visualize states
    try:
        states_list = state_viz.capture_states(input_ids)
        if states_list:
            # Plot state activations
            state_viz.plot_state_activations(
                states_list[0],
                title="Layer 0 State Activations",
                save_path="./visualizations/state_activations.png"
            )
            print("   Saved: state_activations.png")

            # Plot state similarity
            state_viz.plot_state_similarity(
                states_list[-1],
                title="Final Layer State Similarity",
                save_path="./visualizations/state_similarity.png"
            )
            print("   Saved: state_similarity.png")

            # Plot state evolution
            if len(states_list) > 1:
                state_viz.plot_state_evolution(
                    states_list,
                    state_idx=0,
                    save_path="./visualizations/state_evolution.png"
                )
                print("   Saved: state_evolution.png")
    except Exception as e:
        print(f"   State visualization error: {e}")

    # Attention visualization
    print("2. Visualizing attention...")
    attn_viz = AttentionVisualizer(model, figsize=(10, 6))

    try:
        attention_list = attn_viz.capture_attention(input_ids)
        if attention_list:
            # Plot attention heatmap
            attn_viz.plot_attention_heatmap(
                attention_list[0],
                tokens=list(text),
                head_idx=0,
                layer_idx=0,
                save_path="./visualizations/attention_heatmap.png"
            )
            print("   Saved: attention_heatmap.png")

            # Plot all heads
            attn_viz.plot_attention_heads(
                attention_list[0],
                title="Layer 0 Attention Heads",
                save_path="./visualizations/attention_heads.png"
            )
            print("   Saved: attention_heads.png")

            # Plot attention entropy
            attn_viz.plot_attention_entropy(
                attention_list[0],
                save_path="./visualizations/attention_entropy.png"
            )
            print("   Saved: attention_entropy.png")
    except Exception as e:
        print(f"   Attention visualization error: {e}")

    print("\nVisualization complete! Check ./visualizations/ directory.")


if __name__ == "__main__":
    main()
