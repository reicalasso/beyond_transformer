"""
Test configuration for Neural State Machine tests.
"""

import os
import sys
from pathlib import Path
import pytest

# Add src to path
SRC_PATH = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_PATH))

# Test constants
TEST_BATCH_SIZE = 4
TEST_STATE_DIM = 32
TEST_TOKEN_DIM = 16
TEST_NUM_STATES = 8
TEST_SEQ_LEN = 10

# Test directories
TEST_ROOT = Path(__file__).parent
TEST_DATA_DIR = TEST_ROOT / "test_data"
TEST_LOG_DIR = TEST_ROOT / "test_logs"

# Create directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_LOG_DIR.mkdir(exist_ok=True)

# Device configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test tolerances
FLOAT_TOLERANCE = 1e-6
GRADIENT_TOLERANCE = 1e-5

# Performance thresholds
MAX_FORWARD_TIME = 0.1  # seconds
MAX_MEMORY_USAGE = 100 * 1024 * 1024  # 100 MB

# Random seed for reproducibility
RANDOM_SEED = 42

# Test markers
MARKERS = {
    'slow': 'marks tests as slow',
    'integration': 'marks tests as integration tests',
    'gpu': 'marks tests that require GPU',
    'torch': 'marks tests that require PyTorch'
}


def pytest_configure(config):
    """Configure pytest with custom markers."""
    for marker, description in MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_runtest_setup(item):
    """Skip GPU tests if CUDA is not available."""
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# Set random seeds for reproducibility
import torch
import numpy as np
import random

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set deterministic behavior for tests
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False