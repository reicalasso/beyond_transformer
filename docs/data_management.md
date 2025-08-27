# Data Management

This document describes how data is managed in the Beyond Transformer project.

## Overview

The project uses a structured approach to data management with separate directories for raw, processed, and external data.

## Directory Structure

```
data/
├── raw/          # Raw, unprocessed data
├── processed/    # Cleaned and preprocessed data
└── external/     # Data from third-party sources
```

## Supported Datasets

### MNIST

The MNIST dataset is automatically downloaded and processed by the scripts in this project.

- **Raw Data**: Downloaded to `data/raw/`
- **Processed Data**: Saved to `data/processed/`

## Data Downloading

To download all supported datasets, run:

```bash
python scripts/download_data.py
```

This script will:

1. Create the necessary directory structure
2. Download datasets from their official sources
3. Extract compressed files

## Data Preprocessing

To preprocess downloaded data, run:

```bash
python scripts/preprocess_data.py
```

This script will:

1. Load raw data files
2. Apply necessary preprocessing steps (normalization, reshaping, etc.)
3. Save processed data in NumPy format for efficient loading

## Data Loading in Models

To load data in your models, use the data loaders in `src/nsm/data_loaders.py`:

```python
from nsm.data_loaders import get_mnist_dataloaders

# Get data loaders
train_loader, test_loader = get_mnist_dataloaders("data/", batch_size=32)

# Use in training loop
for images, labels in train_loader:
    # Your training code here
    pass
```

## Adding New Datasets

To add a new dataset to the project:

1. **Update `scripts/download_data.py`**:
   - Add download URLs and logic
   - Implement extraction if needed

2. **Update `scripts/preprocess_data.py`**:
   - Add preprocessing logic
   - Save processed data in a consistent format

3. **Update `src/nsm/data_loaders.py`**:
   - Add a new Dataset class if needed
   - Add functions to create data loaders

4. **Update this documentation**:
   - Add information about the new dataset
   - Document any special requirements or considerations

## Data Versioning

For large datasets, we recommend using data versioning tools like:

- [DVC (Data Version Control)](https://dvc.org/)
- [Pachyderm](https://www.pachyderm.com/)
- Cloud storage with versioning (e.g., AWS S3, Google Cloud Storage)

## Best Practices

- **Do not commit large datasets** to the repository
- **Use consistent naming conventions** for data files
- **Document data sources** and licensing information
- **Validate data integrity** after downloading
- **Handle missing or corrupted data** gracefully
- **Use appropriate data types** to minimize memory usage