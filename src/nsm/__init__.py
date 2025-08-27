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

# Core modules
from .modules.state_propagator import StatePropagator
from .modules.state_manager import StateManager as BasicStateManager
from .modules.ssm_block import SSMBlock
from .modules.ntm_memory import NTMMemory
from .modules.transformer_attention import TransformerAttention
from .modules.rnn_memory import RNNMemory

# Layers
from .layers import NSMLayer, HybridAttention

# Components
from .components import TokenToStateRouter, StateManager

# Models
from .models import SimpleNSM

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