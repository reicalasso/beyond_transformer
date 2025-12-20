"""PULSE Core - Modular building blocks for neural architectures."""

# Normalization
from .norm import RMSNorm

# Position Embeddings
from .rope import RotaryEmbedding, apply_rotary_pos_emb

# Attention
from .attention import GroupedQueryAttention, MultiHeadAttention, StateAttention

# Feed-Forward
from .ffn import SwiGLU, GeGLU, MLP

# State Space Model
from .ssm import SelectiveSSM, SSMBlock

# State Management
from .state import GatedStatePropagator, StateManager

# Memory
from .memory import MemoryBank, HierarchicalMemory, StreamingContext

# Spiking & Pulse
from .spiking import SpikingNeuron, PulseProcessor, DynamicRouter, NaturalVariation

# Mixture
from .mixture import MixtureOfExperts, MixtureOfDepths, AdaptiveComputation

# Cache
from .cache import KVCache, CompressedKVCache, SlidingWindowCache

# Speculative
from .speculative import SpeculativeDecoder, DraftModel

__all__ = [
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
