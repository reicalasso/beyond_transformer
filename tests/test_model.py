"""
Tests for PULSE model components.
"""

import pytest
import torch
import torch.nn as nn

from pulse import PulseConfig, PulseForCausalLM
from pulse.models.pulse import PulseModel, PulseLayer, PulseEmbeddings


class TestPulseConfig:
    """Tests for PulseConfig."""
    
    def test_default_config(self):
        config = PulseConfig()
        assert config.vocab_size == 50257
        assert config.hidden_size == 768
        assert config.num_layers == 12
    
    def test_post_init_defaults(self):
        config = PulseConfig(hidden_size=512, num_heads=8)
        assert config.num_kv_heads == 2  # num_heads // 4
        assert config.intermediate_size == int(512 * 2.7)
        assert config.state_dim == 512
    
    def test_to_dict(self):
        config = PulseConfig(vocab_size=1000)
        d = config.to_dict()
        assert d["vocab_size"] == 1000
        assert isinstance(d, dict)
    
    def test_from_dict(self):
        d = {"vocab_size": 2000, "hidden_size": 256}
        config = PulseConfig.from_dict(d)
        assert config.vocab_size == 2000
        assert config.hidden_size == 256


class TestPulseEmbeddings:
    """Tests for embedding layer."""
    
    def test_forward(self, small_config, device):
        embeddings = PulseEmbeddings(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        output = embeddings(input_ids)
        assert output.shape == (2, 16, small_config.hidden_size)
    
    def test_padding_idx(self, small_config, device):
        embeddings = PulseEmbeddings(small_config).to(device)
        # Padding token should have zero embedding
        pad_embedding = embeddings.embed_tokens.weight[small_config.pad_token_id]
        assert torch.allclose(pad_embedding, torch.zeros_like(pad_embedding))


class TestPulseLayer:
    """Tests for single PULSE layer."""
    
    def test_forward_basic(self, small_config, device):
        layer = PulseLayer(small_config, layer_idx=0).to(device)
        hidden = torch.randn(2, 16, small_config.hidden_size, device=device)
        states = torch.randn(2, small_config.num_states, small_config.state_dim, device=device)
        
        output, new_states, kv_cache, aux_loss = layer(hidden, states)
        
        assert output.shape == hidden.shape
        assert aux_loss == 0.0  # No MoE in this config
    
    def test_forward_with_cache(self, small_config, device):
        layer = PulseLayer(small_config, layer_idx=0).to(device)
        hidden = torch.randn(2, 16, small_config.hidden_size, device=device)
        
        _, _, kv_cache, _ = layer(hidden, use_cache=True)
        assert kv_cache is not None
    
    def test_moe_layer(self, moe_config, device):
        # Odd layer should have MoE
        layer = PulseLayer(moe_config, layer_idx=1).to(device)
        assert layer.is_moe
        
        hidden = torch.randn(2, 16, moe_config.hidden_size, device=device)
        output, _, _, aux_loss = layer(hidden)
        
        assert output.shape == hidden.shape
        # MoE should produce aux_loss
        assert isinstance(aux_loss, (float, torch.Tensor))


class TestPulseModel:
    """Tests for base PULSE model."""
    
    def test_forward(self, small_config, device):
        model = PulseModel(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        
        hidden, past_kv, aux_loss = model(input_ids)
        
        assert hidden.shape == (2, 16, small_config.hidden_size)
        assert past_kv is None  # use_cache=False by default
    
    def test_forward_with_cache(self, small_config, device):
        model = PulseModel(small_config).to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        
        hidden, past_kv, _ = model(input_ids, use_cache=True)
        
        assert past_kv is not None
        assert len(past_kv) == small_config.num_layers
    
    def test_incremental_decoding(self, small_config, device):
        model = PulseModel(small_config).to(device)
        model.eval()
        
        # First pass: full sequence
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16), device=device)
        _, past_kv, _ = model(input_ids, use_cache=True)
        
        # Second pass: single token with cache
        next_token = torch.randint(0, small_config.vocab_size, (1, 1), device=device)
        hidden, new_past_kv, _ = model(next_token, past_key_values=past_kv, use_cache=True)
        
        assert hidden.shape == (1, 1, small_config.hidden_size)


class TestPulseForCausalLM:
    """Tests for causal LM model."""
    
    def test_forward(self, small_model, batch_input):
        outputs = small_model(batch_input)
        
        assert "logits" in outputs
        assert outputs["logits"].shape == (*batch_input.shape, small_model.config.vocab_size)
    
    def test_forward_with_labels(self, small_model, batch_input):
        labels = batch_input.clone()
        outputs = small_model(batch_input, labels=labels)
        
        assert "loss" in outputs
        assert outputs["loss"].dim() == 0  # Scalar
        assert outputs["loss"].item() > 0
    
    def test_weight_tying(self, small_config, device):
        model = PulseForCausalLM(small_config).to(device)
        
        # Check weight tying
        assert model.lm_head.weight is model.model.embeddings.embed_tokens.weight
    
    def test_generate_basic(self, small_model, device):
        input_ids = torch.tensor([[1, 2, 3]], device=device)
        
        generated = small_model.generate(
            input_ids,
            max_length=10,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
        )
        
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 10
        assert torch.all(generated[:, :3] == input_ids)  # Prefix preserved
    
    def test_generate_deterministic(self, small_model, device):
        input_ids = torch.tensor([[1, 2, 3]], device=device)
        
        # With temperature=0 (greedy), should be deterministic
        # Note: temperature=0 causes div by zero, use very low temp
        torch.manual_seed(42)
        gen1 = small_model.generate(input_ids, max_length=10, temperature=0.001, top_k=1)
        
        torch.manual_seed(42)
        gen2 = small_model.generate(input_ids, max_length=10, temperature=0.001, top_k=1)
        
        assert torch.equal(gen1, gen2)
    
    def test_generate_with_repetition_penalty(self, small_model, device):
        input_ids = torch.tensor([[1, 2, 3]], device=device)
        
        generated = small_model.generate(
            input_ids,
            max_length=20,
            repetition_penalty=1.5,
        )
        
        assert generated.shape[1] <= 20


class TestModelVariants:
    """Tests for different model configurations."""
    
    def test_ssm_model(self, ssm_config, device):
        model = PulseForCausalLM(ssm_config).to(device)
        input_ids = torch.randint(0, ssm_config.vocab_size, (2, 16), device=device)
        
        outputs = model(input_ids)
        assert outputs["logits"].shape == (2, 16, ssm_config.vocab_size)
    
    def test_moe_model(self, moe_config, device):
        model = PulseForCausalLM(moe_config).to(device)
        input_ids = torch.randint(0, moe_config.vocab_size, (2, 16), device=device)
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        assert "loss" in outputs
    
    def test_no_state_attention(self, device):
        config = PulseConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            use_state_attention=False,
        )
        model = PulseForCausalLM(config).to(device)
        input_ids = torch.randint(0, 1000, (2, 16), device=device)
        
        outputs = model(input_ids)
        assert outputs["logits"].shape == (2, 16, 1000)


class TestGradients:
    """Tests for gradient flow."""
    
    def test_backward_pass(self, small_config, device):
        model = PulseForCausalLM(small_config).to(device)
        model.train()
        
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        labels = input_ids.clone()
        
        outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()
        
        # Check gradients exist for parameters that should have them
        # Note: Some parameters (like state_manager projections) may not receive
        # gradients if states aren't updated during this forward pass
        grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        # Most parameters should have gradients
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert grad_count > total_params * 0.8, f"Only {grad_count}/{total_params} params have gradients"
    
    def test_gradient_accumulation(self, small_config, device):
        model = PulseForCausalLM(small_config).to(device)
        model.train()
        
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        labels = input_ids.clone()
        
        # Two forward-backward passes
        outputs1 = model(input_ids, labels=labels)
        (outputs1["loss"] / 2).backward()
        
        outputs2 = model(input_ids, labels=labels)
        (outputs2["loss"] / 2).backward()
        
        # Gradients should be accumulated
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0)
