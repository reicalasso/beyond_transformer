"""Modular building blocks for the modern PULSE architecture."""

from .block import AttentionBlock, DeltaBlock
from .conv import ShortCausalConv1d
from .delta import DeltaState, GatedDeltaRule
from .ffn import SwiGLU
from .norm import RMSNorm, l2_normalize
from .rope import RotaryEmbedding, apply_rope
from .swa import SlidingWindowAttention, SWACache

__all__ = [
    "AttentionBlock",
    "DeltaBlock",
    "DeltaState",
    "GatedDeltaRule",
    "RMSNorm",
    "RotaryEmbedding",
    "SWACache",
    "ShortCausalConv1d",
    "SlidingWindowAttention",
    "SwiGLU",
    "apply_rope",
    "l2_normalize",
]
