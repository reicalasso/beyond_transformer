"""
Test script for Hybrid Models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from nsm.models.hybrid_model import AdvancedHybridModel, SequentialHybridModel


def test_advanced_hybrid_model():
    """Test AdvancedHybridModel."""
    print("Testing AdvancedHybridModel...")
    
    # Create model with custom config
    config = {
        'input_dim': 784,
        'output_dim': 10,
        'embedding_dim': 64,
        'sequence_length': 8,
        'ssm_dim': 64,
        'ntm_mem_size': 64,
        'ntm_mem_dim': 16,
        'rnn_hidden_dim': 64,
        'attention_heads': 8
    }
    
    model = AdvancedHybridModel(config)
    
    # Check model info
    info = model.get_model_info()
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 784)
    output = model(x)
    
    assert output.shape == (batch_size, 10), f"Output shape mismatch: {output.shape}"
    print(f"  Input {x.shape} → Output {output.shape} ✓")
    
    # Test attention weights
    attn_weights = model.get_attention_weights(x)
    expected_attn_shape = (batch_size, config['attention_heads'], config['sequence_length'], config['sequence_length'])
    assert attn_weights.shape == expected_attn_shape, f"Attention weights shape mismatch: {attn_weights.shape}"
    print(f"  Attention weights {attn_weights.shape} ✓")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("  Backward pass ✓")
    
    print("✓ AdvancedHybridModel test passed\n")


def test_sequential_hybrid_model():
    """Test SequentialHybridModel."""
    print("Testing SequentialHybridModel...")
    
    # Create model
    model = SequentialHybridModel(input_dim=784, output_dim=10)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 784)
    output = model(x)
    
    assert output.shape == (batch_size, 10), f"Output shape mismatch: {output.shape}"
    print(f"  Input {x.shape} → Output {output.shape} ✓")
    
    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("  Backward pass ✓")
    
    print("✓ SequentialHybridModel test passed\n")


def test_model_comparison():
    """Compare both models."""
    print("Comparing models...")
    
    # Create both models
    advanced_config = {'input_dim': 784, 'output_dim': 10, 'embedding_dim': 32, 'sequence_length': 4}
    advanced_model = AdvancedHybridModel(advanced_config)
    sequential_model = SequentialHybridModel(input_dim=784, output_dim=10)
    
    # Parameter count comparison
    advanced_params = sum(p.numel() for p in advanced_model.parameters())
    sequential_params = sum(p.numel() for p in sequential_model.parameters())
    
    print(f"  AdvancedHybridModel parameters: {advanced_params:,}")
    print(f"  SequentialHybridModel parameters: {sequential_params:,}")
    
    # Test with same input
    batch_size = 2
    x = torch.randn(batch_size, 784)
    
    advanced_output = advanced_model(x)
    sequential_output = sequential_model(x)
    
    print(f"  Advanced output shape: {advanced_output.shape}")
    print(f"  Sequential output shape: {sequential_output.shape}")
    
    print("✓ Model comparison completed\n")


def run_all_tests():
    """Run all tests."""
    print("Running Hybrid Model Tests")
    print("=" * 40)
    
    try:
        test_advanced_hybrid_model()
        test_sequential_hybrid_model()
        test_model_comparison()
        
        print("=" * 40)
        print("🎉 All hybrid model tests passed!")
        return True
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)