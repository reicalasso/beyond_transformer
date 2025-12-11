Installation
============

System Requirements
-------------------

Parallel Unified Linear State Engines requires:

* Python 3.8 or higher
* PyTorch 2.0.0 or higher
* CUDA 11.0 or higher (for GPU acceleration)
* At least 4GB of RAM for basic usage
* 16GB+ RAM recommended for large-scale experiments

Installation Methods
--------------------

Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install beyond-transformer

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   git clone https://github.com/reicalasso/beyond_transformer.git
   cd beyond_transformer
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

   git clone https://github.com/reicalasso/beyond_transformer.git
   cd beyond_transformer
   pip install -e ".[dev]"

This installs additional development dependencies including:

* ``black`` for code formatting
* ``flake8`` for linting
* ``mypy`` for type checking
* ``pytest`` for testing
* ``sphinx`` for documentation

GPU Support
-----------

pulses support CUDA acceleration out of the box. Ensure you have:

1. NVIDIA GPU with CUDA Compute Capability 3.5 or higher
2. CUDA toolkit installed
3. PyTorch with CUDA support

To verify GPU availability:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")

Docker Installation
-------------------

A Docker image is available for easy deployment:

.. code-block:: bash

   docker pull reicalasso/beyond-transformer:latest
   docker run -it --gpus all reicalasso/beyond-transformer:latest

Verification
------------

Verify your installation:

.. code-block:: python

   import pulse
   from pulse import Simplepulse
   import torch
   
   # Create a simple model
   model = Simplepulse(vocab_size=1000, d_model=128, num_states=32)
   
   # Test forward pass
   x = torch.randint(0, 1000, (2, 100))
   output = model(x)
   
   print(f"Model output shape: {output.shape}")
   print("Installation successful!")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'pulse'**

Solution: Ensure the package is installed correctly:

.. code-block:: bash

   pip list | grep beyond-transformer

**CUDA out of memory**

Solution: Reduce batch size or model dimensions:

.. code-block:: python

   model = Simplepulse(
       vocab_size=10000,
       d_model=256,    # Reduce from 512
       num_states=32,  # Reduce from 64
   )

**Slow training on CPU**

Solution: Enable GPU acceleration or use smaller models for CPU:

.. code-block:: python

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)

Getting Help
~~~~~~~~~~~~

* GitHub Issues: https://github.com/reicalasso/beyond_transformer/issues
* Documentation: https://beyond-transformer.readthedocs.io
* Email: reicalasso@gmail.com