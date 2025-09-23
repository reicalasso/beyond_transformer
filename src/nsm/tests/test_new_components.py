"""
Simple test for new NSM components.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from nsm.modules.ntm_memory import NTMMemory
from nsm.modules.rnn_memory import RNNMemory
from nsm.modules.ssm_block import SSMBlock
from nsm.modules.transformer_attention import TransformerAttention


def test_components():
    """Test all new components."""
    print("Testing NSM Core Components...")

    # Test SSMBlock
    print("\n1. Testing SSMBlock...")
    ssm = SSMBlock(d_model=64, d_state=16, d_conv=4, expand=2)
    x = torch.randn(2, 10, 64)
    out = ssm(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    print("   ✓ SSMBlock OK")

    # Test NTMMemory
    print("\n2. Testing NTMMemory...")
    ntm = NTMMemory(mem_size=128, mem_dim=20, num_read_heads=1, num_write_heads=1)
    read_keys = torch.randn(2, 1, 20)
    write_keys = torch.randn(2, 1, 20)
    read_strengths = torch.randn(2, 1)
    write_strengths = torch.randn(2, 1)
    erase_vectors = torch.randn(2, 1, 20)
    add_vectors = torch.randn(2, 1, 20)

    read_vecs, mem_state = ntm(
        read_keys,
        write_keys,
        read_strengths,
        write_strengths,
        erase_vectors,
        add_vectors,
    )
    print(f"   Read vectors: {read_vecs.shape}")
    print(f"   Memory state: {mem_state.shape}")
    assert read_vecs.shape == (2, 1, 20)
    assert mem_state.shape == (128, 20)
    print("   ✓ NTMMemory OK")

    # Test TransformerAttention
    print("\n3. Testing TransformerAttention...")
    attn = TransformerAttention(d_model=64, num_heads=8)
    x = torch.randn(2, 10, 64)
    out, weights = attn.forward_self_attention(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Attention weights: {weights.shape}")
    assert out.shape == x.shape
    assert weights.shape == (2, 8, 10, 10)
    print("   ✓ TransformerAttention OK")

    # Test RNNMemory
    print("\n4. Testing RNNMemory...")
    rnn = RNNMemory(input_dim=64, hidden_dim=128, num_layers=2, rnn_type="gru")
    x = torch.randn(2, 10, 64)
    hidden = rnn.init_hidden(2, torch.device("cpu"))
    out, hidden_out = rnn(x, hidden)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   Hidden state: {type(hidden_out)}")
    assert out.shape == (2, 10, 128)
    print("   ✓ RNNMemory OK")

    print("\n🎉 All components tested successfully!")


if __name__ == "__main__":
    test_components()
