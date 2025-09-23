"""
Smoke test for NSM components integration
"""

import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsm import NSMLayer, StateManager, TokenToStateRouter


def test_component_integration():
    """Test that all components work together."""
    print("Running NSM Components Integration Smoke Test")
    print("=" * 50)

    # Test parameters
    batch_size = 2
    seq_len = 10
    token_dim = 32
    state_dim = 64
    num_states = 8

    # Create components
    print("1. Creating components...")
    router = TokenToStateRouter(token_dim, state_dim, num_states)
    nsm_layer = NSMLayer(state_dim=state_dim, token_dim=state_dim, num_heads=4)
    state_manager = StateManager(state_dim=state_dim, max_states=16, initial_states=8)

    # Create test data
    print("2. Creating test data...")
    tokens = torch.randn(batch_size, seq_len, token_dim)
    initial_states = state_manager()  # Get initial states
    states = initial_states.unsqueeze(0).repeat(
        batch_size, 1, 1
    )  # [batch_size, num_states, state_dim]

    print(f"   Tokens shape: {tokens.shape}")
    print(f"   States shape: {states.shape}")

    # Test TokenToStateRouter
    print("3. Testing TokenToStateRouter...")
    routed_tokens, routing_weights = router(tokens, states)
    print(f"   Routed tokens shape: {routed_tokens.shape}")
    print(f"   Routing weights shape: {routing_weights.shape}")

    # Verify routing weights sum to 1
    weight_sums = routing_weights.sum(dim=-1)
    assert torch.allclose(
        weight_sums, torch.ones_like(weight_sums)
    ), "Routing weights should sum to 1"
    print("   ✓ Routing weights sum to 1")

    # Test NSMLayer
    print("4. Testing NSMLayer...")
    updated_states = nsm_layer(states, routed_tokens)
    print(f"   Updated states shape: {updated_states.shape}")

    # Verify output shape
    assert updated_states.shape == states.shape, "Output shape should match input shape"
    print("   ✓ Output shape is correct")

    # Test StateManager
    print("5. Testing StateManager...")
    current_states = state_manager()
    print(f"   Current states shape: {current_states.shape}")
    print(f"   Active states count: {state_manager.get_active_count()}")

    # Test state info
    state_info = state_manager.get_state_info()
    print(f"   State info keys: {list(state_info.keys())}")

    # Test pruning
    pruned = state_manager.prune_low_importance_states()
    print(f"   States pruned: {pruned}")

    # Test allocation
    allocated = state_manager.allocate_states(2)
    print(f"   States allocated: {allocated}")

    # Differentiability test (without state management operations)
    print("6. Testing differentiability...")
    tokens.requires_grad_(True)

    # Create fresh states for gradient test to avoid issues with state management
    fresh_states = torch.randn(batch_size, num_states, state_dim, requires_grad=True)

    routed_tokens, _ = router(tokens, fresh_states)
    updated_states = nsm_layer(fresh_states, routed_tokens)
    loss = updated_states.sum()
    loss.backward()

    assert tokens.grad is not None, "Tokens should have gradients"
    assert fresh_states.grad is not None, "States should have gradients"
    print("   ✓ Components are differentiable")

    print("\n✓ All integration tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_component_integration()
        print("\n🎉 Smoke test completed successfully!")
    except Exception as e:
        print(f"\n❌ Smoke test failed with error: {e}")
        raise
