"""
PULSE - Parallel Unified Linear State Engine

Minimal, efficient, production-ready sequence backbone.

Core ideas:
- UnifiedBlock: O(n) local + global processing
- SimpleMemory: LRU cache for persistence
- RecurrentState: Single compressed state
"""

__version__ = "3.0.0"

# Models
from .models import (
    # Current PULSE model
    PulseConfig,
    PulseForCausalLM,
    PulseModel,
    # Explicit v2 implementation (kept for compatibility)
    PulseV2Config,
    PulseV2ForCausalLM,
    PulseV2,
    # Legacy v1 implementation
    LegacyPulseConfig,
    LegacyPulseForCausalLM,
    LegacyPulseModel,
)

from .core import (
    # Normalization
    RMSNorm,
    # Position encodings
    RotaryEmbedding,
    apply_rotary_pos_emb,
    # Attention
    GroupedQueryAttention,
    MultiHeadAttention,
    # Feed-forward
    SwiGLU,
    # Unified processing
    UnifiedBlock,
    LinearAttention,
    LocalConv,
    RecurrentState,
    # External memory
    KeyValueMemory,
    MemoryAugmentedLayer,
    # Backwards-compatible names
    SimpleMemory,
    MemoryAugmentedBlock,
)

__all__ = [
    # Current PULSE model
    "PulseConfig",
    "PulseForCausalLM",
    "PulseModel",
    # Explicit v2 (compatibility)
    "PulseV2Config",
    "PulseV2ForCausalLM",
    "PulseV2",
    # Legacy v1
    "LegacyPulseConfig",
    "LegacyPulseForCausalLM",
    "LegacyPulseModel",
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
    # Unified processing
    "UnifiedBlock",
    "LinearAttention",
    "LocalConv",
    "RecurrentState",
    # External memory
    "KeyValueMemory",
    "MemoryAugmentedLayer",
    # Backwards-compatible names
    "SimpleMemory",
    "MemoryAugmentedBlock",
]
