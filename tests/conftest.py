"""
Pytest configuration and shared fixtures for PULSE tests.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse import PulseConfig, PulseForCausalLM


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return PulseConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_states=8,
        state_dim=64,
        intermediate_size=128,
        max_position_embeddings=256,
        dropout=0.0,
    )


@pytest.fixture
def small_model(small_config, device):
    """Small model for testing."""
    model = PulseForCausalLM(small_config).to(device)
    model.eval()
    return model


@pytest.fixture
def batch_input(small_config, device):
    """Sample batch input."""
    batch_size = 2
    seq_len = 32
    return torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device=device)


@pytest.fixture
def moe_config():
    """Config with MoE enabled."""
    return PulseConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        num_states=8,
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        dropout=0.0,
    )


@pytest.fixture
def ssm_config():
    """Config with SSM enabled."""
    return PulseConfig(
        vocab_size=1000,
        hidden_size=64,
        num_layers=3,
        num_heads=4,
        num_states=8,
        use_ssm=True,
        dropout=0.0,
    )
