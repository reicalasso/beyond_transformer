"""
Data preprocessing utilities.
"""

import os
import numpy as np
import torch
from pathlib import Path


def preprocess_mnist(data_dir):
    """
    Preprocess MNIST data.
    
    Args:
        data_dir (str): Directory containing the raw data
    """
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Process training data
    train_images_file = os.path.join(raw_dir, "train-images-idx3-ubyte")
    train_labels_file = os.path.join(raw_dir, "train-labels-idx1-ubyte")
    
    # Load and preprocess training images
    with open(train_images_file, 'rb') as f:
        # Skip header (16 bytes)
        f.read(16)
        # Read data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_samples, 28, 28)
        train_images = data.reshape(-1, 28, 28)
    
    # Load training labels
    with open(train_labels_file, 'rb') as f:
        # Skip header (8 bytes)
        f.read(8)
        # Read data
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Normalize images to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    
    # Save processed training data
    np.save(os.path.join(processed_dir, "train_images.npy"), train_images)
    np.save(os.path.join(processed_dir, "train_labels.npy"), train_labels)
    
    # Process test data
    test_images_file = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
    test_labels_file = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
    
    # Load and preprocess test images
    with open(test_images_file, 'rb') as f:
        # Skip header (16 bytes)
        f.read(16)
        # Read data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape to (num_samples, 28, 28)
        test_images = data.reshape(-1, 28, 28)
    
    # Load test labels
    with open(test_labels_file, 'rb') as f:
        # Skip header (8 bytes)
        f.read(8)
        # Read data
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Normalize images to [0, 1]
    test_images = test_images.astype(np.float32) / 255.0
    
    # Save processed test data
    np.save(os.path.join(processed_dir, "test_images.npy"), test_images)
    np.save(os.path.join(processed_dir, "test_labels.npy"), test_labels)
    
    print(f"Processed data saved to {processed_dir}")


def load_processed_mnist(data_dir):
    """
    Load preprocessed MNIST data.
    
    Args:
        data_dir (str): Directory containing the processed data
        
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    processed_dir = os.path.join(data_dir, "processed")
    
    # Load processed data
    train_images = np.load(os.path.join(processed_dir, "train_images.npy"))
    train_labels = np.load(os.path.join(processed_dir, "train_labels.npy"))
    test_images = np.load(os.path.join(processed_dir, "test_images.npy"))
    test_labels = np.load(os.path.join(processed_dir, "test_labels.npy"))
    
    return train_images, train_labels, test_images, test_labels


# Example usage
if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = os.path.join(project_root, "data")
    
    # Preprocess data
    preprocess_mnist(data_dir)
    
    # Load processed data
    train_images, train_labels, test_images, test_labels = load_processed_mnist(data_dir)
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    print("Data preprocessing test completed!")