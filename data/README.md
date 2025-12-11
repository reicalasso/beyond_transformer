# Data Management

This directory is intended for storing datasets used in experiments.

## Structure

- `raw/`: Raw, unprocessed data
- `processed/`: Cleaned and preprocessed data
- `external/`: Data from third-party sources

## Supported Datasets

### MNIST

The MNIST dataset is automatically downloaded and processed by the scripts in this project.

- **Raw Data**: Downloaded to `data/raw/`
- **Processed Data**: Saved to `data/processed/`

### Adding New Datasets

To add a new dataset:

1. Add download logic to `scripts/download_data.py`
2. Add preprocessing logic to `scripts/preprocess_data.py`
3. Add data loading utilities to `src//data_loaders.py`

## Data Versioning

For large datasets, we recommend using data versioning tools like:

- [DVC (Data Version Control)](https://dvc.org/)
- [Pachyderm](https://www.pachyderm.com/)
- Cloud storage with versioning (e.g., AWS S3, Google Cloud Storage)

## Usage

### Downloading Data

To download all supported datasets:

```bash
python scripts/download_data.py
```

### Preprocessing Data

To preprocess downloaded data:

```bash
python scripts/preprocess_data.py
```

### Loading Data in Models

To load data in your models, use the data loaders in `src//data_loaders.py`:

```python
from pulse.data_loaders import get_mnist_dataloaders

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders("data/", batch_size=32)

# Use in training loop
for images, labels in train_loader:
    # Your training code here
    pass
```

## Note

Do not commit large datasets to the repository. Use data versioning tools or store data in cloud storage with scripts to download.