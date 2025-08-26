"""
Neural State Machine (NSM) Package
==============================

A PyTorch implementation of Neural State Machines as an alternative to Transformers.

Modules:
- modules: Core NSM components including StatePropagator
- models: Complete NSM model implementations
- utils: Utility functions and helpers
- experiments: Experimental implementations and research code

Example:
    >>> from nsm.modules import StatePropagator
    >>> propagator = StatePropagator(state_dim=128, gate_type='gru')
    >>> # Use propagator in your model
"""

__version__ = "0.1.0"
__author__ = "Beyond Transformer Team"

# Import main modules
from .modules import StatePropagator

__all__ = [
    "StatePropagator",
]