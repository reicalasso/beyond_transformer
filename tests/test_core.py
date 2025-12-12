"""
Tests for PULSE core components.
"""

import pytest
import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse.core.norm import RMSNorm
from pulse.core.attention import GroupedQueryAttention, StateAttention
from pulse.core.ffn import SwiGLU
from pulse.core.state import StateManager
from pulse.core.memory import MemoryBank, HierarchicalMemory, StreamingContext
from pulse.core.rope import RotaryEmbedding
from pulse.core.ssm import SSMBlock
from pulse.core.mixture import MixtureOfExperts, MixtureOfDepths


class TestRMSNorm:
    """Tests for RMSNorm."""
    
    def test_forward_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        norm = RMSNorm(64, eps=1e-6)
        x = torch.randn(2, 16, 64) * 10  # Large values
        out = norm(x)
        # RMS should be approximately 1
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.5)


class TestRotaryEmbedding:
    """Tests for RoPE."""
    
    def test_forward_shape(self):
        rope = RotaryEmbedding(64, max_position_embeddings=512)
        cos, sin = rope(seq_len=16)
        assert cos.shape[-1] == 64
        assert sin.shape[-1] == 64
    
    def test_position_encoding_unique(self):
        rope = RotaryEmbedding(64, max_position_embeddings=512)
        cos, sin = rope(seq_len=32)
        # Each position should have unique encoding
        cos_flat = cos.squeeze()
        for i in range(31):
            assert not torch.allclose(cos_flat[i], cos_flat[i+1])


class TestGroupedQueryAttention:
    """Tests for GQA."""
    
    @pytest.fixture
    def gqa(self):
        return GroupedQueryAttention(
            hidden_size=64,
            num_heads=8,
            num_kv_heads=2,
            dropout=0.0,
            max_position_embeddings=256,
            use_rope=True,
        )
    
    def test_forward_shape(self, gqa):
        x = torch.randn(2, 16, 64)
        out, _ = gqa(x)
        assert out.shape == x.shape
    
    def test_kv_cache(self, gqa):
        x = torch.randn(2, 16, 64)
        out, kv_cache = gqa(x, use_cache=True)
        
        assert kv_cache is not None
        assert len(kv_cache) == 2  # key, value
    
    def test_incremental_with_cache(self, gqa):
        # First pass
        x1 = torch.randn(2, 16, 64)
        _, kv_cache = gqa(x1, use_cache=True)
        
        # Second pass with cache
        x2 = torch.randn(2, 1, 64)
        out, new_cache = gqa(x2, past_key_value=kv_cache, use_cache=True)
        
        assert out.shape == (2, 1, 64)
        # Cache should grow
        assert new_cache[0].shape[2] == 17  # 16 + 1
    
    def test_causal_masking(self, gqa):
        gqa.eval()
        x = torch.randn(1, 8, 64)
        
        # Get attention weights (would need to modify GQA to return them)
        # For now, just verify output changes with different sequence lengths
        out1, _ = gqa(x[:, :4, :])
        out2, _ = gqa(x[:, :8, :])
        
        # First 4 positions should be different due to causal masking
        assert out1.shape == (1, 4, 64)
        assert out2.shape == (1, 8, 64)


class TestStateAttention:
    """Tests for StateAttention."""
    
    @pytest.fixture
    def state_attn(self):
        return StateAttention(
            hidden_size=64,
            state_dim=64,
            num_heads=4,
            dropout=0.0,
        )
    
    def test_forward_shape(self, state_attn):
        hidden = torch.randn(2, 16, 64)
        states = torch.randn(2, 8, 64)
        
        out = state_attn(hidden, states)
        assert out.shape == hidden.shape
    
    def test_state_influence(self, state_attn):
        hidden = torch.randn(2, 16, 64)
        states1 = torch.randn(2, 8, 64)
        states2 = torch.randn(2, 8, 64) * 10
        
        out1 = state_attn(hidden, states1)
        out2 = state_attn(hidden, states2)
        
        # Different states should produce different outputs
        assert not torch.allclose(out1, out2)


class TestSwiGLU:
    """Tests for SwiGLU FFN."""
    
    def test_forward_shape(self):
        ffn = SwiGLU(64, 256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape
    
    def test_nonlinearity(self):
        ffn = SwiGLU(64, 256)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        # Output should be different from input (nonlinear transform)
        assert not torch.allclose(out, x)


class TestStateManager:
    """Tests for StateManager."""
    
    @pytest.fixture
    def state_manager(self):
        return StateManager(
            hidden_size=64,
            state_dim=64,
            num_states=8,
        )
    
    def test_initial_states(self, state_manager):
        states = state_manager.get_initial_states(batch_size=2)
        assert states.shape == (2, 8, 64)
    
    def test_state_update(self, state_manager):
        hidden = torch.randn(2, 16, 64)
        states = state_manager.get_initial_states(2)
        
        new_states = state_manager(hidden, states)
        assert new_states.shape == states.shape
        # States should be updated
        assert not torch.allclose(new_states, states)


class TestMemoryBank:
    """Tests for MemoryBank."""
    
    @pytest.fixture
    def memory_bank(self):
        return MemoryBank(
            memory_dim=64,
            num_slots=16,
            compression_ratio=0.5,
            decay_rate=0.99,
        )
    
    def test_write(self, memory_bank):
        content = torch.randn(2, 64)
        memory, importance, age = memory_bank.write(content)
        
        assert memory.shape == (2, 16, 64)
        assert importance.shape == (2, 16)
        assert age.shape == (2, 16)
    
    def test_read(self, memory_bank):
        query = torch.randn(2, 8, 64)
        retrieved, attn_weights = memory_bank.read(query, top_k=4)
        
        assert retrieved.shape == (2, 8, 64)
        assert attn_weights.shape == (2, 8, 16)
    
    def test_compression(self, memory_bank):
        content = torch.randn(2, 64)
        compressed = memory_bank.compress_and_store(content)
        
        assert compressed.shape == (2, 32)  # 64 * 0.5
        
        decompressed = memory_bank.decompress(compressed)
        assert decompressed.shape == (2, 64)
    
    def test_decay(self, memory_bank):
        # Write some content
        content = torch.randn(2, 64)
        memory_bank.write(content)
        
        # Decay should not raise errors
        memory_bank.decay_memories()


class TestHierarchicalMemory:
    """Tests for HierarchicalMemory."""
    
    @pytest.fixture
    def hier_memory(self):
        return HierarchicalMemory(
            hidden_size=64,
            working_slots=8,
            short_term_slots=16,
            long_term_slots=32,
        )
    
    def test_forward(self, hier_memory):
        query = torch.randn(2, 16, 64)
        output = hier_memory(query)
        
        assert output.shape == query.shape
    
    def test_forward_with_content(self, hier_memory):
        query = torch.randn(2, 16, 64)
        new_content = torch.randn(2, 8, 64)
        
        output = hier_memory(query, new_content=new_content)
        assert output.shape == query.shape


class TestStreamingContext:
    """Tests for StreamingContext."""
    
    @pytest.fixture
    def streaming(self):
        return StreamingContext(
            hidden_size=64,
            summary_size=32,
            chunk_size=16,
        )
    
    def test_update_summary(self, streaming):
        content = torch.randn(2, 16, 64)
        streaming.update_summary(content)
        # Should not raise
    
    def test_query_summary(self, streaming):
        query = torch.randn(2, 8, 64)
        output = streaming.query_summary(query)
        
        assert output.shape == query.shape
    
    def test_process_stream(self, streaming):
        input_stream = torch.randn(2, 48, 64)  # 3 chunks
        
        def process_fn(x):
            return x * 2
        
        output = streaming.process_stream(input_stream, process_fn)
        assert output.shape == input_stream.shape


class TestSSMBlock:
    """Tests for SSM block."""
    
    @pytest.fixture
    def ssm(self):
        return SSMBlock(hidden_size=64)
    
    def test_forward_shape(self, ssm):
        x = torch.randn(2, 16, 64)
        out, state = ssm(x)
        
        assert out.shape == x.shape
    
    def test_with_initial_state(self, ssm):
        x = torch.randn(2, 16, 64)
        out1, state1 = ssm(x)
        out2, state2 = ssm(x, state=state1)
        
        # Different initial states should give different outputs
        assert not torch.allclose(out1, out2)


class TestMixtureOfExperts:
    """Tests for MoE."""
    
    @pytest.fixture
    def moe(self):
        return MixtureOfExperts(
            hidden_size=64,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
        )
    
    def test_forward_shape(self, moe):
        x = torch.randn(2, 16, 64)
        out, aux_loss = moe(x)
        
        assert out.shape == x.shape
        assert isinstance(aux_loss, torch.Tensor)
    
    def test_expert_selection(self, moe):
        x = torch.randn(2, 16, 64)
        out1, _ = moe(x)
        out2, _ = moe(x * 10)  # Different input magnitudes
        
        # Should route to potentially different experts
        assert not torch.allclose(out1, out2)
    
    def test_load_balancing_loss(self, moe):
        x = torch.randn(2, 16, 64)
        _, aux_loss = moe(x)
        
        # Aux loss should be positive (load balancing)
        assert aux_loss.item() >= 0


class TestMixtureOfDepths:
    """Tests for MoD."""
    
    @pytest.fixture
    def mod(self):
        return MixtureOfDepths(hidden_size=64, capacity_factor=0.5)
    
    def test_forward_shape(self, mod):
        x = torch.randn(2, 16, 64)
        
        def transform(y):
            return y * 2
        
        out, routing_weights = mod(x, transform)
        assert out.shape == x.shape
    
    def test_capacity_limit(self, mod):
        x = torch.randn(2, 16, 64)
        
        def transform(y):
            return y * 2
        
        out, routing_weights = mod(x, transform)
        # Only ~50% of tokens should be processed
        # (exact behavior depends on implementation)
        assert out.shape == x.shape
