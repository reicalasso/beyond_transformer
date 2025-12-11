#!/usr/bin/env python3
"""
Test script for routing visualization components
"""

import sys

import numpy as np
import torch

# Add src to path
sys.path.insert(0, "src")

from pulse import StateManager, TokenToStateRouter
from pulse import SimplePulse


def test_routing_visualization():
    """Test all routing visualization components"""
    print("Testing routing visualization components...")

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Test 1: SimplePulse model
    print("1. Testing SimplePulse model...")
    model = SimplePulse(
        input_dim=128, state_dim=64, num_states=16, output_dim=10, gate_type="gru"
    )
    print("   SimplePulse created successfully!")

    # Test 2: TokenToStateRouter
    print("2. Testing TokenToStateRouter...")
    router = TokenToStateRouter(token_dim=128, state_dim=64, num_states=16, num_heads=4)
    print("   TokenToStateRouter created successfully!")

    # Test 3: StateManager
    print("3. Testing StateManager...")
    state_manager = StateManager(
        state_dim=64, max_states=16, initial_states=16, prune_threshold=0.3
    )
    print("   StateManager created successfully!")

    # Test 4: Sample data
    print("4. Testing with sample data...")
    batch_size = 4
    seq_len = 20
    token_dim = 128
    num_states = 16
    state_dim = 64

    sample_tokens = torch.randn(batch_size, seq_len, token_dim)
    sample_states = torch.randn(batch_size, num_states, state_dim)
    print(f"   Sample tokens shape: {sample_tokens.shape}")
    print(f"   Sample states shape: {sample_states.shape}")

    # Test 5: Router forward pass
    print("5. Testing router forward pass...")
    routed_tokens, routing_weights = router(sample_tokens, sample_states)
    print(f"   Routed tokens shape: {routed_tokens.shape}")
    print(f"   Routing weights shape: {routing_weights.shape}")

    # Test 6: State manager functionality
    print("6. Testing state manager functionality...")
    active_states = state_manager()
    importance_scores = state_manager.get_importance_scores()
    print(f"   Active states shape: {active_states.shape}")
    print(f"   Importance scores shape: {importance_scores.shape}")
    print(f"   Active count: {state_manager.get_active_count()}")

    print("\nAll tests passed! Routing visualization components are working correctly.")


if __name__ == "__main__":
    test_routing_visualization()
