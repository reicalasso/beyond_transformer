"""Shared pytest fixtures and helpers for the PULSE test suite."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Seed everything per test for reproducibility."""
    seed = 1234
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # too aggressive for some ops


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assert_close(actual: torch.Tensor, expected: torch.Tensor, **kw) -> None:
    """Wrapper around torch.testing.assert_close with sensible defaults."""
    kw.setdefault("rtol", 1e-4)
    kw.setdefault("atol", 1e-5)
    torch.testing.assert_close(actual, expected, **kw)
