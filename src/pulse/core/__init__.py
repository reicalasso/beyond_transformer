"""
PULSE core modules: reusable building blocks for sequence models.
"""

# Normalization
from .norm import RMSNorm

# Position Embeddings
from .rope import RotaryEmbedding, apply_rotary_pos_emb

# Attention (minimal)
from .attention import GroupedQueryAttention, MultiHeadAttention

# Feed-Forward (minimal)
from .ffn import SwiGLU

# Unified processing block
from .unified import UnifiedBlock, LinearAttention, LocalConv, RecurrentState

# External memory
from .memory import KeyValueMemory, MemoryAugmentedLayer, SimpleMemory, MemoryAugmentedBlock

__all__ = [
    # Norm
    "RMSNorm",
    # Position
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    # Attention
    "GroupedQueryAttention",
    "MultiHeadAttention",
    # FFN
    "SwiGLU",
    # Unified (v2)
    "UnifiedBlock",
    "LinearAttention",
    "LocalConv",
    "RecurrentState",
    # Memory
    "KeyValueMemory",
    "MemoryAugmentedLayer",
    # Backwards-compatible names
    "SimpleMemory",
    "MemoryAugmentedBlock",
]
