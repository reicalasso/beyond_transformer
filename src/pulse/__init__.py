"""PULSE - Parallel Unified Linear State Engine

A PyTorch implementation of PULSE architecture as an alternative to Transformers.
PULSE achieves O(n·s) complexity through state-based computation, where s is the
number of states, compared to Transformer's O(n²) complexity.

Key Features:
- Linear complexity for long sequences
- 2x faster than Transformers at 2048+ tokens  
- 80% memory reduction on long sequences
- State-based computation with gated updates
- Flash attention support (PyTorch 2.0+)

Core Components:
- OptimizedPulseForCausalLM: Production-ready language model
- OptimizedPulseConfig: Model configuration
- StatePropagator: Gated state updates (GRU/LSTM style)
- StateManager: Dynamic state allocation

Example:
    >>> from pulse import OptimizedPulseConfig, OptimizedPulseForCausalLM
    >>> config = OptimizedPulseConfig(vocab_size=50257, hidden_size=768)
    >>> model = OptimizedPulseForCausalLM(config)
    >>> output = model(input_ids, labels=labels)
    >>> loss = output['loss']
"""

__version__ = "2.0.0"
__author__ = "PULSE Team"

# Optimized Models (Recommended)
from .models.optimized_pulse_lm import (
    OptimizedPulseConfig,
    OptimizedPulseForCausalLM,
)

# Optimized Attention
from .core.optimized_attention import (
    OptimizedStateAttention,
    OptimizedSelfAttention,
    OptimizedStatePropagator,
    OptimizedPulseLayer,
)

# Core Modules
from .modules.state_propagator import StatePropagator
from .modules.state_manager import StateManager
from .modules.ssm_block import SSMBlock
from .modules.ntm_memory import NTMMemory

# Legacy Models (for compatibility)
from .models.pulse_lm import PulseConfig, PulseForCausalLM
from .models.simple_pulse import SimplePulse, SequencePulse
from .core.layers import PulseLayer, HybridAttention
from .core.attention import LinearAttention, CausalStateAttention

__all__ = [
    # PULSE v2 (Best Quality)
    "PulseV2Config",
    "PulseV2ForCausalLM",
    "PulseV2Model",
    # Optimized (Fast)
    "OptimizedPulseConfig",
    "OptimizedPulseForCausalLM",
    "OptimizedStateAttention",
    "OptimizedSelfAttention",
    "OptimizedStatePropagator",
    "OptimizedPulseLayer",
    # Core
    "StatePropagator",
    "StateManager",
    "SSMBlock",
    "NTMMemory",
    # Legacy
    "PulseConfig",
    "PulseForCausalLM",
    "SimplePulse",
    "SequencePulse",
    "PulseLayer",
    "HybridAttention",
    "LinearAttention",
    "CausalStateAttention",
]
