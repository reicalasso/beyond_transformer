"""
PULSE Models - Language models built on PULSE architecture.

v2: Minimal, efficient, powerful.
"""

# New (v2) - recommended
from .pulse_v2 import (
    PulseV2Config,
    PulseV2ForCausalLM,
    PulseV2,
)

# Legacy (v1) - kept for compatibility
from .pulse_legacy import (
    PulseConfig,
    PulseForCausalLM,
    PulseModel,
)

__all__ = [
    # v2 (recommended)
    "PulseV2Config",
    "PulseV2ForCausalLM",
    "PulseV2",
    # v1 (legacy)
    "PulseConfig",
    "PulseForCausalLM",
    "PulseModel",
]
