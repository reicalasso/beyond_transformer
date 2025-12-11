"""
Data loading utilities.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MNISTDataset(Dataset):
    """
    MNIST Dataset loader.
    """

    def __init__(self, data_dir, train=True, transform=None):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Directory containing the data
            train (bool): If True, load training data, otherwise test data
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        # Load data
        self._load_data()

    def _load_data(self):
        """Load MNIST data from files."""
        raw_dir = os.path.join(self.data_dir, "raw")

        if self.train:
            images_file = os.path.join(raw_dir, "train-images-idx3-ubyte")
            labels_file = os.path.join(raw_dir, "train-labels-idx1-ubyte")
        else:
            images_file = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
            labels_file = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")

        # Load images
        with open(images_file, "rb") as f:
            # Skip header (16 bytes)
            f.read(16)
            # Read data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            # Reshape to (num_samples, 28, 28)
            self.images = data.reshape(-1, 28, 28)

        # Load labels
        with open(labels_file, "rb") as f:
            # Skip header (8 bytes)
            f.read(8)
            # Read data
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def __len__(self):
        """Return the number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label) where image is a tensor and label is an int
        """
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor
        image = torch.from_numpy(image).float()
        # Normalize to [0, 1]
        image = image / 255.0
        # Flatten the image
        image = image.view(-1)

        # Convert label to Python int
        label = int(label)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_mnist_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    Get MNIST data loaders.

    Args:
        data_dir (str): Directory containing the data
        batch_size (int): Batch size
        num_workers (int): Number of worker processes

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MNISTDataset(data_dir, train=True)
    test_dataset = MNISTDataset(data_dir, train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = os.path.join(project_root, "data")

    # Get data loaders
    train_loader, test_loader = get_mnist_dataloaders(data_dir, batch_size=32)

    # Print dataset info
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # Print first batch
    for images, labels in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch labels shape: {labels.shape}")
        break

    print("Data loading test completed!")
