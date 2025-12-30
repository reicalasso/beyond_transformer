"""
PULSE Core - Minimal building blocks for neural architectures.

v2: Radically simplified. Keep it simple & efficient.
"""

# Normalization
from .norm import RMSNorm

# Position Embeddings
from .rope import RotaryEmbedding, apply_rotary_pos_emb

# Attention (minimal)
from .attention import GroupedQueryAttention, MultiHeadAttention

# Feed-Forward (minimal)
from .ffn import SwiGLU

# Unified Block (NEW - replaces SSM, State, Mixture complexity)
from .unified import UnifiedBlock, LinearAttention, LocalConv, RecurrentState

# Simple Memory (NEW - replaces HierarchicalMemory)
from .simple_memory import SimpleMemory, MemoryAugmentedBlock

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
    # Memory (v2)
    "SimpleMemory",
    "MemoryAugmentedBlock",
]
