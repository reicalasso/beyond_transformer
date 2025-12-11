"""
PULSE Command Line Interface.

This module provides CLI tools for:
- Training models
- Running inference
- Benchmarking
- Model conversion
"""

from .main import main, app

__all__ = ["main", "app"]
