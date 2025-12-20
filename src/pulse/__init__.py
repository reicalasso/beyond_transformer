"""PULSE - Parallel Unified Linear State Engine

A biologically-inspired neural architecture with hierarchical memory.
"""

__version__ = "1.0.0"

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
