"""
PULSE - Parallel Unified Linear State Engine

v2: Radically simplified. Keep it simple & efficient.

A minimal neural architecture with:
- UnifiedBlock: O(n) local+global processing
- SimpleMemory: LRU cache for persistence
- RecurrentState: Single compressed state
"""

__version__ = "2.0.0"

# Models (v2 recommended)
from .models import (
    # v2 - recommended
    PulseV2Config,
    PulseV2ForCausalLM,
    PulseV2,
    # v1 - legacy
    PulseConfig,
    PulseForCausalLM,
    PulseModel,
)

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
    # FFN
    SwiGLU,
    # Unified (v2)
    UnifiedBlock,
    LinearAttention,
    LocalConv,
    RecurrentState,
    # Memory (v2)
    SimpleMemory,
    MemoryAugmentedBlock,
)

__all__ = [
    # Models (v2)
    "PulseV2Config",
    "PulseV2ForCausalLM",
    "PulseV2",
    # Models (v1 legacy)
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
