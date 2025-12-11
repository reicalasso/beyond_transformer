"""
Tests for PULSE components and layers
"""

import torch

from pulse import HybridAttention, PulseLayer, StateManager, TokenToStateRouter


def test_pulse_layer():
    """Test PulseLayer functionality."""
    batch_size, num_states, state_dim = 2, 8, 128
    seq_len, token_dim = 10, 64

    # Create layer
    layer = PulseLayer(state_dim=state_dim, token_dim=token_dim, num_heads=4)

    # Create sample inputs
    states = torch.randn(batch_size, num_states, state_dim)
    tokens = torch.randn(batch_size, seq_len, token_dim)

    # Forward pass
    updated_states = layer(states, tokens)

    assert updated_states.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "PulseLayer output shape mismatch"
    print("✓ PulseLayer test passed")


def test_hybrid_attention():
    """Test HybridAttention functionality."""
    batch_size, num_states, state_dim = 2, 8, 128
    seq_len, token_dim = 10, 64

    # Create attention mechanism
    attention = HybridAttention(state_dim=state_dim, token_dim=token_dim, num_heads=4)

    # Create sample inputs
    states = torch.randn(batch_size, num_states, state_dim)
    tokens = torch.randn(batch_size, seq_len, token_dim)

    # Forward pass
    attended_tokens = attention(states, tokens)

    assert attended_tokens.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "HybridAttention output shape mismatch"
    print("✓ HybridAttention test passed")


def test_token_to_state_router():
    """Test TokenToStateRouter functionality."""
    batch_size, seq_len, token_dim = 2, 10, 64
    num_states, state_dim = 8, 128

    # Create router
    router = TokenToStateRouter(token_dim, state_dim, num_states)

    # Create sample inputs
    tokens = torch.randn(batch_size, seq_len, token_dim)
    states = torch.randn(batch_size, num_states, state_dim)

    # Forward pass
    routed_tokens, routing_weights = router(tokens, states)

    assert routed_tokens.shape == (
        batch_size,
        num_states,
        state_dim,
    ), "Routed tokens shape mismatch"
    assert routing_weights.shape == (
        batch_size,
        seq_len,
        num_states,
    ), "Routing weights shape mismatch"
    print("✓ TokenToStateRouter test passed")


def test_state_manager():
    """Test StateManager functionality."""
    state_dim = 128
    max_states = 16
    initial_states = 8

    # Create state manager
    manager = StateManager(
        state_dim=state_dim, max_states=max_states, initial_states=initial_states
    )

    # Test initial states
    active_states = manager()
    assert (
        active_states.shape[0] == initial_states
    ), "Initial active states count mismatch"
    assert active_states.shape[1] == state_dim, "State dimension mismatch"

    # Test allocation
    allocated = manager.allocate_states(3)
    assert allocated == 3, "Allocation count mismatch"
    assert (
        manager.get_active_count() == initial_states + allocated
    ), "Active count after allocation mismatch"

    # Test pruning
    pruned = manager.prune_low_importance_states()
    assert pruned >= 0, "Pruning should return non-negative count"

    print("✓ StateManager test passed")


def run_all_tests():
    """Run all component tests."""
    print("Running PULSE Component Tests...")
    print("=" * 40)

    test_pulse_layer()
    test_hybrid_attention()
    test_token_to_state_router()
    test_state_manager()

    print("\n✓ All component tests passed!")


if __name__ == "__main__":
    run_all_tests()
