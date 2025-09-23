"""
Tests for data loading utilities.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsm.data.data_loaders import MNISTDataset, get_mnist_dataloaders


def test_mnist_dataset():
    """Test MNISTDataset functionality."""
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_dir = os.path.join(temp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create dummy MNIST files
        # Create dummy train images file
        train_images_file = os.path.join(raw_dir, "train-images-idx3-ubyte")
        # Create a simple header (16 bytes) + dummy data
        with open(train_images_file, "wb") as f:
            # Write dummy header (16 bytes)
            f.write(b"\x00\x00\x08\x03\x00\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00\x1c")
            # Write dummy image data (2 images of 28x28)
            dummy_data = np.random.randint(0, 256, size=(2, 28, 28), dtype=np.uint8)
            f.write(dummy_data.tobytes())

        # Create dummy train labels file
        train_labels_file = os.path.join(raw_dir, "train-labels-idx1-ubyte")
        with open(train_labels_file, "wb") as f:
            # Write dummy header (8 bytes)
            f.write(b"\x00\x00\x08\x01\x00\x00\x00\x02")
            # Write dummy labels
            f.write(np.array([3, 7], dtype=np.uint8).tobytes())

        # Create dummy test files
        test_images_file = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
        with open(test_images_file, "wb") as f:
            # Write dummy header (16 bytes)
            f.write(b"\x00\x00\x08\x03\x00\x00\x00\x01\x00\x00\x00\x1c\x00\x00\x00\x1c")
            # Write dummy image data (1 image of 28x28)
            dummy_data = np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8)
            f.write(dummy_data.tobytes())

        test_labels_file = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
        with open(test_labels_file, "wb") as f:
            # Write dummy header (8 bytes)
            f.write(b"\x00\x00\x08\x01\x00\x00\x00\x01")
            # Write dummy labels
            f.write(np.array([5], dtype=np.uint8).tobytes())

        # Test training dataset
        train_dataset = MNISTDataset(temp_dir, train=True)
        assert len(train_dataset) == 2, "Training dataset size mismatch"

        # Test test dataset
        test_dataset = MNISTDataset(temp_dir, train=False)
        assert len(test_dataset) == 1, "Test dataset size mismatch"

        # Test data loading
        image, label = train_dataset[0]
        assert image.shape == (784,), "Image shape mismatch"
        assert isinstance(label, int), "Label type mismatch"

        print("✓ MNISTDataset test passed")


def test_get_mnist_dataloaders():
    """Test get_mnist_dataloaders functionality."""
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_dir = os.path.join(temp_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create dummy MNIST files (same as above)
        # Create dummy train images file
        train_images_file = os.path.join(raw_dir, "train-images-idx3-ubyte")
        with open(train_images_file, "wb") as f:
            # Write dummy header (16 bytes)
            f.write(b"\x00\x00\x08\x03\x00\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00\x1c")
            # Write dummy image data (2 images of 28x28)
            dummy_data = np.random.randint(0, 256, size=(2, 28, 28), dtype=np.uint8)
            f.write(dummy_data.tobytes())

        # Create dummy train labels file
        train_labels_file = os.path.join(raw_dir, "train-labels-idx1-ubyte")
        with open(train_labels_file, "wb") as f:
            # Write dummy header (8 bytes)
            f.write(b"\x00\x00\x08\x01\x00\x00\x00\x02")
            # Write dummy labels
            f.write(np.array([3, 7], dtype=np.uint8).tobytes())

        # Create dummy test files
        test_images_file = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
        with open(test_images_file, "wb") as f:
            # Write dummy header (16 bytes)
            f.write(b"\x00\x00\x08\x03\x00\x00\x00\x01\x00\x00\x00\x1c\x00\x00\x00\x1c")
            # Write dummy image data (1 image of 28x28)
            dummy_data = np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8)
            f.write(dummy_data.tobytes())

        test_labels_file = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
        with open(test_labels_file, "wb") as f:
            # Write dummy header (8 bytes)
            f.write(b"\x00\x00\x08\x01\x00\x00\x00\x01")
            # Write dummy labels
            f.write(np.array([5], dtype=np.uint8).tobytes())

        # Get data loaders
        train_loader, test_loader = get_mnist_dataloaders(temp_dir, batch_size=2)

        # Test train loader
        assert len(train_loader) == 1, "Train loader batch count mismatch"

        # Test test loader
        assert len(test_loader) == 1, "Test loader batch count mismatch"

        # Test data loading
        for images, labels in train_loader:
            assert images.shape == (2, 784), "Train batch image shape mismatch"
            assert labels.shape == (2,), "Train batch label shape mismatch"
            break

        for images, labels in test_loader:
            assert images.shape == (1, 784), "Test batch image shape mismatch"
            assert labels.shape == (1,), "Test batch label shape mismatch"
            break

        print("✓ get_mnist_dataloaders test passed")


def run_all_tests():
    """Run all data loading tests."""
    print("Running Data Loading Tests...")
    print("=" * 30)

    test_mnist_dataset()
    test_get_mnist_dataloaders()

    print("\n✓ All data loading tests passed!")


if __name__ == "__main__":
    run_all_tests()
