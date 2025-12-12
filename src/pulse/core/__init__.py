"""
PULSE Core - Modular building blocks.

Each module has a single responsibility:
- norm.py: RMSNorm, LayerNorm
- rope.py: Rotary Position Embeddings
- attention.py: GQA, MHA, StateAttention
- ffn.py: SwiGLU, GeGLU, MLP
- ssm.py: Selective State Space Model
- state.py: State propagation and management
- memory.py: Hierarchical memory
- spiking.py: Pulse processing, dynamic routing
- mixture.py: MoE, MoD, ACT
- cache.py: KV cache variants
- speculative.py: Speculative decoding
"""

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
