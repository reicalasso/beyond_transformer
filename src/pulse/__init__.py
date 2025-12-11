"""
PULSE - Parallel Unified Linear State Engine

A PyTorch implementation of PULSE architecture as an alternative to Transformers.
PULSE achieves linear complexity O(n) through state-based computation.

Main Components:
- StatePropagator: Core state update mechanisms with gating
- PulseLayer: Complete PULSE layer with hybrid attention
- StateManager: Dynamic state allocation and pruning
- TokenToStateRouter: Routing mechanism for tokens to states
- SSMBlock: State Space Model layer
- NTMMemory: Neural Turing Machine memory
- TransformerAttention: Standard Transformer attention
- RNNMemory: RNN-based memory layer

Language Models:
- PulseConfig: Configuration for PULSE language models
- PulseForCausalLM: Autoregressive language model
- PulseForSequenceClassification: Sequence classification
- PulseForTokenClassification: Token classification (NER, POS)

Training:
- PulseTrainer: Training loop with logging and checkpointing
- TrainingArguments: Training configuration

Example:
    >>> from pulse import StatePropagator, PulseLayer, StateManager
    >>> propagator = StatePropagator(state_dim=128, gate_type='gru')
    >>> layer = PulseLayer(state_dim=128, token_dim=64)
    >>> manager = StateManager(state_dim=128, max_states=32)
    
    >>> # Language modeling
    >>> from pulse import PulseConfig, PulseForCausalLM
    >>> config = PulseConfig(vocab_size=32000, hidden_size=768)
    >>> model = PulseForCausalLM(config)
"""

__version__ = "1.0.0"
__author__ = "PULSE Team"

# Components
from .modules.state_manager import StateManager
from .core.components import TokenToStateRouter

# Layers
from .core.layers import HybridAttention, PulseLayer

# Models - Basic
from .models import SequencePulse, SimplePulse
from .modules.ntm_memory import NTMMemory
from .modules.rnn_memory import RNNMemory
from .modules.ssm_block import SSMBlock
from .modules.state_manager import StateManager as BasicStateManager

# Core modules
from .modules.state_propagator import StatePropagator
from .modules.transformer_attention import TransformerAttention

# Language Models
from .models.pulse_lm import (
    PulseConfig,
    PulseForCausalLM,
    PulseForSequenceClassification,
    PulseForTokenClassification,
)

# Advanced attention mechanisms
from .core.attention import (
    SparseStateAttention,
    LinearAttention,
    CausalStateAttention,
    MultiScaleAttention,
)

# Adaptive state management
from .core.adaptive_state import (
    AdaptiveStateAllocator,
    StateCompressor,
    HierarchicalStateManager,
)

__all__ = [
    # Core
    "StatePropagator",
    "SSMBlock",
    "NTMMemory",
    "TransformerAttention",
    "RNNMemory",
    "PulseLayer",
    "HybridAttention",
    "TokenToStateRouter",
    "StateManager",
    "BasicStateManager",
    # Basic Models
    "SimplePulse",
    "SequencePulse",
    # Language Models
    "PulseConfig",
    "PulseForCausalLM",
    "PulseForSequenceClassification",
    "PulseForTokenClassification",
    # Advanced Attention
    "SparseStateAttention",
    "LinearAttention",
    "CausalStateAttention",
    "MultiScaleAttention",
    # Adaptive State
    "AdaptiveStateAllocator",
    "StateCompressor",
    "HierarchicalStateManager",
]
