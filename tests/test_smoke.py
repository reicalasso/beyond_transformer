"""
Simple smoke test to verify test framework.
"""


def test_framework_works():
    """Test that the test framework is working."""
    assert True


def test_pytest_available():
    """Test that pytest is available."""
    try:
        import pytest

        assert pytest is not None
    except ImportError:
        assert False, "pytest not available"


def test_torch_available():
    """Test that PyTorch is available."""
    try:
        import torch

        assert torch is not None
    except ImportError:
        assert False, "PyTorch not available"


def test_basic_math():
    """Test basic mathematical operations."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
    assert 10 / 2 == 5


def test_torch_tensor_creation():
    """Test PyTorch tensor creation."""
    import torch

    tensor = torch.randn(2, 3)
    assert tensor.shape == (2, 3)
    assert torch.isfinite(tensor).all()


if __name__ == "__main__":
    # Run simple tests
    test_framework_works()
    test_pytest_available()
    test_torch_available()
    test_basic_math()
    test_torch_tensor_creation()
    print("✅ All smoke tests passed!")
