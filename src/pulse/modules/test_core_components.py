"""
Integration tests for PULSE core components.

This module tests the integration of all core components:
- SSMBlock
- NTMMemory
- TransformerAttention
- RNNMemory
"""

import torch

from pulse.modules.ntm_memory import NTMMemory
from pulse.modules.rnn_memory import RNNMemory
from pulse.modules.ssm_block import SSMBlock
from pulse.modules.transformer_attention import TransformerAttention


def test_ssm_block():
    """Test SSMBlock functionality."""
    print("Testing SSMBlock...")

    batch_size, seq_len, d_model = 2, 10, 64

    # Create SSMBlock
    ssm_block = SSMBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ssm_block(x)

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"SSMBlock output shape mismatch: {output.shape}"
    print("✓ SSMBlock test passed")


def test_ntm_memory():
    """Test NTMMemory functionality."""
    print("Testing NTMMemory...")

    batch_size, mem_size, mem_dim = 2, 128, 20
    num_read_heads, num_write_heads = 1, 1

    # Create NTMMemory
    ntm_memory = NTMMemory(
        mem_size=mem_size,
        mem_dim=mem_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
    )

    # Create sample inputs
    read_keys = torch.randn(batch_size, num_read_heads, mem_dim)
    write_keys = torch.randn(batch_size, num_write_heads, mem_dim)
    read_strengths = torch.randn(batch_size, num_read_heads)
    write_strengths = torch.randn(batch_size, num_write_heads)
    erase_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
    add_vectors = torch.randn(batch_size, num_write_heads, mem_dim)

    # Forward pass
    read_vectors, memory_state = ntm_memory(
        read_keys,
        write_keys,
        read_strengths,
        write_strengths,
        erase_vectors,
        add_vectors,
    )

    assert read_vectors.shape == (
        batch_size,
        num_read_heads,
        mem_dim,
    ), f"NTM read vectors shape mismatch: {read_vectors.shape}"
    assert memory_state.shape == (
        mem_size,
        mem_dim,
    ), f"NTM memory state shape mismatch: {memory_state.shape}"
    print("✓ NTMMemory test passed")


def test_transformer_attention():
    """Test TransformerAttention functionality."""
    print("Testing TransformerAttention...")

    batch_size, seq_len, d_model = 2, 10, 64
    num_heads = 8

    # Create TransformerAttention
    transformer_attn = TransformerAttention(d_model=d_model, num_heads=num_heads)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass (self-attention)
    output, attn_weights = transformer_attn.forward_self_attention(x)

    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Transformer output shape mismatch: {output.shape}"
    assert attn_weights.shape == (
        batch_size,
        num_heads,
        seq_len,
        seq_len,
    ), f"Attention weights shape mismatch: {attn_weights.shape}"
    print("✓ TransformerAttention test passed")


def test_rnn_memory():
    """Test RNNMemory functionality."""
    print("Testing RNNMemory...")

    batch_size, seq_len, input_dim, hidden_dim = 2, 10, 64, 128
    num_layers = 2

    # Test LSTM
    lstm_memory = RNNMemory(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="lstm",
    )

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    lstm_memory = lstm_memory.to(device)

    # Initialize hidden state
    hidden = lstm_memory.init_hidden(batch_size, device)

    # Forward pass
    output, hidden_state = lstm_memory(x, hidden)

    assert output.shape == (
        batch_size,
        seq_len,
        hidden_dim,
    ), f"LSTM output shape mismatch: {output.shape}"
    assert (
        isinstance(hidden_state, tuple) and len(hidden_state) == 2
    ), "LSTM hidden state should be a tuple of (h, c)"
    print("✓ RNNMemory (LSTM) test passed")

    # Test GRU
    gru_memory = RNNMemory(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="gru",
    )
    gru_memory = gru_memory.to(device)

    # Initialize hidden state
    hidden = gru_memory.init_hidden(batch_size, device)

    # Forward pass
    output, hidden_state = gru_memory(x, hidden)

    assert output.shape == (
        batch_size,
        seq_len,
        hidden_dim,
    ), f"GRU output shape mismatch: {output.shape}"
    assert isinstance(hidden_state, torch.Tensor), "GRU hidden state should be a tensor"
    print("✓ RNNMemory (GRU) test passed")


def run_all_tests():
    """Run all component tests."""
    print("Running PULSE Core Component Integration Tests...")
    print("=" * 50)

    try:
        test_ssm_block()
        test_ntm_memory()
        test_transformer_attention()
        test_rnn_memory()

        print("\n" + "=" * 50)
        print("✓ All PULSE core component tests passed!")
        return True
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
