"""
Data downloading and preprocessing scripts.
"""

import os
import sys
import urllib.request
import zipfile
import gzip
import shutil
from pathlib import Path


def download_file(url, filename):
    """
    Download a file from a URL.
    
    Args:
        url (str): URL to download from
        filename (str): Local filename to save to
    """
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded to {filename}")


def extract_zip(zip_path, extract_to):
    """
    Extract a ZIP file.
    
    Args:
        zip_path (str): Path to the ZIP file
        extract_to (str): Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def extract_gz(gz_path, extract_to):
    """
    Extract a GZ file.
    
    Args:
        gz_path (str): Path to the GZ file
        extract_to (str): Path to extract to (without .gz extension)
    """
    print(f"Extracting {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(extract_to, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Extracted to {extract_to}")


def download_mnist(data_dir):
    """
    Download and extract MNIST dataset.
    
    Args:
        data_dir (str): Directory to save data to
    """
    # Create directories
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # MNIST URLs
    mnist_urls = {
        "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    }
    
    # Download files
    for filename, url in mnist_urls.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            download_file(url, filepath)
        
        # Extract if it's a .gz file
        if filename.endswith(".gz"):
            extract_to = os.path.join(raw_dir, filename[:-3])  # Remove .gz extension
            if not os.path.exists(extract_to):
                extract_gz(filepath, extract_to)


def main():
    """Main function to download all datasets."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = os.path.join(project_root, "data")
    
    print("Downloading datasets...")
    print(f"Data directory: {data_dir}")
    
    # Download MNIST
    download_mnist(data_dir)
    
    print("All datasets downloaded and extracted!")


if __name__ == "__main__":
    main()