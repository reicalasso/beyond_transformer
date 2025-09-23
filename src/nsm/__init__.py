"""
Neural State Machine (NSM) Package

A PyTorch implementation of Neural State Machines as an alternative to Transformers.

Main Components:
- StatePropagator: Core state update mechanisms with gating
- NSMLayer: Complete NSM layer with hybrid attention
- StateManager: Dynamic state allocation and pruning
- TokenToStateRouter: Routing mechanism for tokens to states
- SSMBlock: State Space Model layer
- NTMMemory: Neural Turing Machine memory
- TransformerAttention: Standard Transformer attention
- RNNMemory: RNN-based memory layer

Example:
    >>> from nsm import StatePropagator, NSMLayer, StateManager
    >>> propagator = StatePropagator(state_dim=128, gate_type='gru')
    >>> layer = NSMLayer(state_dim=128, token_dim=64)
    >>> manager = StateManager(state_dim=128, max_states=32)
"""

__version__ = "0.1.0"
__author__ = "Beyond Transformer Team"

# Components
from .modules.state_manager import StateManager
from .core.components import TokenToStateRouter

# Layers
from .core.layers import HybridAttention, NSMLayer

# Models
from .models import SimpleNSM
from .modules.ntm_memory import NTMMemory
from .modules.rnn_memory import RNNMemory
from .modules.ssm_block import SSMBlock
from .modules.state_manager import StateManager as BasicStateManager

# Core modules
from .modules.state_propagator import StatePropagator
from .modules.transformer_attention import TransformerAttention

__all__ = [
    "StatePropagator",
    "SSMBlock",
    "NTMMemory",
    "TransformerAttention",
    "RNNMemory",
    "NSMLayer",
    "HybridAttention",
    "TokenToStateRouter",
    "StateManager",
    "BasicStateManager",
    "SimpleNSM",
]
