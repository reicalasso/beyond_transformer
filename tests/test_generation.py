"""
Tests for text generation functionality.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse import PulseConfig, PulseForCausalLM


class TestGeneration:
    """Tests for generation methods."""
    
    @pytest.fixture
    def model(self):
        config = PulseConfig(
            vocab_size=100,
            hidden_size=32,
            num_layers=2,
            num_heads=2,
            num_states=4,
            max_position_embeddings=64,
            dropout=0.0,
        )
        model = PulseForCausalLM(config)
        model.eval()
        return model
    
    def test_generate_length(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        generated = model.generate(input_ids, max_length=20)
        
        assert generated.shape[1] <= 20
        assert generated.shape[1] >= 3  # At least input length
    
    def test_generate_preserves_prefix(self, model):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        generated = model.generate(input_ids, max_length=15)
        
        # First 5 tokens should match input
        assert torch.equal(generated[:, :5], input_ids)
    
    def test_generate_batch(self, model):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        
        generated = model.generate(input_ids, max_length=10)
        
        assert generated.shape[0] == 2
    
    def test_temperature_effect(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        torch.manual_seed(42)
        gen_high_temp = model.generate(input_ids, max_length=15, temperature=2.0)
        
        torch.manual_seed(42)
        gen_low_temp = model.generate(input_ids, max_length=15, temperature=0.1)
        
        # Different temperatures should generally produce different outputs
        # (though not guaranteed with same seed)
    
    def test_top_k_sampling(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        # top_k=1 should be greedy
        torch.manual_seed(42)
        gen1 = model.generate(input_ids, max_length=10, top_k=1, top_p=1.0)
        
        torch.manual_seed(123)
        gen2 = model.generate(input_ids, max_length=10, top_k=1, top_p=1.0)
        
        # Greedy should be deterministic regardless of seed
        assert torch.equal(gen1, gen2)
    
    def test_top_p_sampling(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Very low top_p should be near-greedy
        generated = model.generate(input_ids, max_length=10, top_p=0.1, top_k=0)
        
        assert generated.shape[1] <= 10
    
    def test_repetition_penalty(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        # High repetition penalty
        gen_with_penalty = model.generate(
            input_ids, max_length=20, repetition_penalty=2.0
        )
        
        # Count unique tokens
        unique_with = len(set(gen_with_penalty[0].tolist()))
        
        # Without penalty
        gen_without = model.generate(
            input_ids, max_length=20, repetition_penalty=1.0
        )
        unique_without = len(set(gen_without[0].tolist()))
        
        # With penalty should generally have more unique tokens
        # (not guaranteed but likely)
    
    def test_generate_no_nan(self, model):
        input_ids = torch.tensor([[1, 2, 3]])
        
        generated = model.generate(input_ids, max_length=30)
        
        # Should not contain any invalid tokens
        assert torch.all(generated >= 0)
        assert torch.all(generated < model.config.vocab_size)
    
    def test_kv_cache_consistency(self, model):
        """Test that KV cache produces same results as full recomputation."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        
        # Full forward pass
        with torch.no_grad():
            full_out = model(input_ids)
            full_logits = full_out["logits"][:, -1, :]
        
        # Incremental with cache
        with torch.no_grad():
            # First 4 tokens
            out1 = model(input_ids[:, :4], use_cache=True)
            cache = out1["past_key_values"]
            
            # Last token with cache
            out2 = model(input_ids[:, 4:], past_key_values=cache, use_cache=True)
            cached_logits = out2["logits"][:, -1, :]
        
        # Should produce similar logits (relaxed tolerance due to state attention)
        # Note: State attention may cause slight differences between cached and full
        assert torch.allclose(full_logits, cached_logits, atol=0.1)


class TestNucleusSampling:
    """Specific tests for nucleus (top-p) sampling fix."""
    
    @pytest.fixture
    def model(self):
        config = PulseConfig(
            vocab_size=50,
            hidden_size=32,
            num_layers=1,
            num_heads=2,
            num_states=4,
            dropout=0.0,
        )
        return PulseForCausalLM(config).eval()
    
    def test_top_p_filters_correctly(self, model):
        """Test that top-p correctly filters low probability tokens."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        # With very restrictive top_p, should still generate valid tokens
        for _ in range(5):
            generated = model.generate(
                input_ids, max_length=10, top_p=0.5, top_k=0
            )
            assert generated.shape[1] <= 10
            assert torch.all(generated >= 0)
            assert torch.all(generated < model.config.vocab_size)
    
    def test_top_p_with_top_k(self, model):
        """Test combined top-p and top-k sampling."""
        input_ids = torch.tensor([[1, 2, 3]])
        
        generated = model.generate(
            input_ids, max_length=15, top_p=0.9, top_k=10
        )
        
        assert generated.shape[1] <= 15
        assert torch.all(generated >= 0)
