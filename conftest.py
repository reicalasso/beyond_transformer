"""
pytest configuration and fixtures for Neural State Machine tests.
"""

import sys
import os
import torch
import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set deterministic behavior for tests
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.fixture
def device():
    """Get the appropriate device for testing (CUDA if available, otherwise CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_batch_size():
    """Sample batch size for testing."""
    return 4


@pytest.fixture
def sample_state_dim():
    """Sample state dimension for testing."""
    return 32


@pytest.fixture
def sample_token_dim():
    """Sample token dimension for testing."""
    return 16


@pytest.fixture
def sample_num_states():
    """Sample number of states for testing."""
    return 8


@pytest.fixture
def sample_seq_len():
    """Sample sequence length for testing."""
    return 10


@pytest.fixture
def sample_input_tensor(sample_batch_size, sample_token_dim):
    """Create a sample input tensor."""
    return torch.randn(sample_batch_size, sample_token_dim)


@pytest.fixture
def sample_state_tensor(sample_batch_size, sample_state_dim):
    """Create a sample state tensor."""
    return torch.randn(sample_batch_size, sample_state_dim)


@pytest.fixture
def sample_multi_state_tensor(sample_batch_size, sample_num_states, sample_state_dim):
    """Create a sample multi-state tensor."""
    return torch.randn(sample_batch_size, sample_num_states, sample_state_dim)


@pytest.fixture
def sample_sequence_tensor(sample_batch_size, sample_seq_len, sample_token_dim):
    """Create a sample sequence tensor."""
    return torch.randn(sample_batch_size, sample_seq_len, sample_token_dim)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture(autouse=True)
def cleanup_tensors():
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "torch: marks tests that require PyTorch")


def pytest_runtest_setup(item):
    """Skip GPU tests if CUDA is not available."""
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")