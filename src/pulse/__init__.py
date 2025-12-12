"""PULSE - Parallel Unified Linear State Engine

A modular neural architecture with standardized building blocks.

Structure:
    pulse/
    ├── core/           # Building blocks
    │   ├── norm.py     # RMSNorm
    │   ├── rope.py     # Rotary Position Embeddings
    │   ├── attention.py # GQA, MHA, StateAttention
    │   ├── ffn.py      # SwiGLU, GeGLU, MLP
    │   ├── ssm.py      # Selective State Space Model
    │   ├── state.py    # State propagation
    │   ├── memory.py   # Hierarchical memory
    │   ├── spiking.py  # Pulse processing
    │   ├── mixture.py  # MoE, MoD, ACT
    │   ├── cache.py    # KV cache variants
    │   └── speculative.py # Speculative decoding
    └── models/
        └── pulse.py    # Main model

Quick Start:
    >>> from pulse import PulseConfig, PulseForCausalLM
    >>> config = PulseConfig(vocab_size=50257, hidden_size=768)
    >>> model = PulseForCausalLM(config)
    >>> output = model(input_ids, labels=labels)
"""

__version__ = "4.0.0"
__author__ = "PULSE Team"

# Models
from .models import PulseConfig, PulseForCausalLM, PulseModel

# Core building blocks
from .core import (
    # Norm
    RMSNorm,
    # Position
    RotaryEmbedding,
    apply_rotary_pos_emb,
    # Attention
    GroupedQueryAttention,
    MultiHeadAttention,
    StateAttention,
    # FFN
    SwiGLU,
    GeGLU,
    MLP,
    # SSM
    SelectiveSSM,
    SSMBlock,
    # State
    GatedStatePropagator,
    StateManager,
    # Memory
    MemoryBank,
    HierarchicalMemory,
    StreamingContext,
    # Spiking
    SpikingNeuron,
    PulseProcessor,
    DynamicRouter,
    NaturalVariation,
    # Mixture
    MixtureOfExperts,
    MixtureOfDepths,
    AdaptiveComputation,
    # Cache
    KVCache,
    CompressedKVCache,
    SlidingWindowCache,
    # Speculative
    SpeculativeDecoder,
    DraftModel,
)

__all__ = [
    # Models
    "PulseConfig",
    "PulseForCausalLM",
    "PulseModel",
    # Norm
    "RMSNorm",
    # Position
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    # Attention
    "GroupedQueryAttention",
    "MultiHeadAttention",
    "StateAttention",
    # FFN
    "SwiGLU",
    "GeGLU",
    "MLP",
    # SSM
    "SelectiveSSM",
    "SSMBlock",
    # State
    "GatedStatePropagator",
    "StateManager",
    # Memory
    "MemoryBank",
    "HierarchicalMemory",
    "StreamingContext",
    # Spiking
    "SpikingNeuron",
    "PulseProcessor",
    "DynamicRouter",
    "NaturalVariation",
    # Mixture
    "MixtureOfExperts",
    "MixtureOfDepths",
    "AdaptiveComputation",
    # Cache
    "KVCache",
    "CompressedKVCache",
    "SlidingWindowCache",
    # Speculative
    "SpeculativeDecoder",
    "DraftModel",
]
