"""
Neural State Machine (NSM) Package

A PyTorch implementation of Neural State Machines as an alternative to Transformers.

Main Components:
- StatePropagator: Core state update mechanisms with gating
- NSMLayer: Complete NSM layer with hybrid attention
- StateManager: Dynamic state allocation and pruning
- TokenToStateRouter: Routing mechanism for tokens to states

Example:
    >>> from nsm import StatePropagator, NSMLayer, StateManager
    >>> propagator = StatePropagator(state_dim=128, gate_type='gru')
    >>> layer = NSMLayer(state_dim=128, token_dim=64)
    >>> manager = StateManager(state_dim=128, max_states=32)
"""

__version__ = "0.1.0"
__author__ = "Beyond Transformer Team"

# Core modules
from .modules.state_propagator import StatePropagator
from .modules.state_manager import StateManager as BasicStateManager

# Layers
from .layers import NSMLayer, HybridAttention

# Components
from .components import TokenToStateRouter, StateManager

# Models
from .models import SimpleNSM

__all__ = [
    "StatePropagator",
    "NSMLayer", 
    "HybridAttention",
    "TokenToStateRouter",
    "StateManager",
    "BasicStateManager",
    "SimpleNSM",
]