"""
Setup script for Neural State Machine (NSM) package.
"""

from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="beyond-transformer",
    version="0.1.0",
    author="Beyond Transformer Team",
    author_email="your.email@example.com",
    description="Neural State Machines as an alternative to Transformer architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/beyond_transformer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "psutil>=5.8.0",
        "pandas>=1.3.0",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "jupyter>=1.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "experiments": [
            "scikit-learn>=1.0.0",
            "tensorboard>=2.5.0",
            "wandb>=0.12.0",
        ],
        "test": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nsm-train=scripts.train_model:main",
            "nsm-evaluate=scripts.evaluate_model:main",
            "nsm-visualize=scripts.visualize_results:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)