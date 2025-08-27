from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Ana bağımlılıklar
with open("requirements/requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Test bağımlılıkları
with open("requirements/requirements-test.txt", "r", encoding="utf-8") as f:
    test_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Deney bağımlılıkları
with open("requirements/requirements-experiments.txt", "r", encoding="utf-8") as f:
    experiment_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="beyond_transformer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A research project exploring alternatives to the Transformer architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/beyond_transformer", # Değiştirin
    project_urls={
        "Bug Reports": "https://github.com/yourusername/beyond_transformer/issues", # Değiştirin
        "Source": "https://github.com/yourusername/beyond_transformer", # Değiştirin
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
        "experiment": experiment_requires,
    },
)