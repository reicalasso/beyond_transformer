"""
PULSE models built on the core architecture.

`PulseConfig` / `PulseModel` / `PulseForCausalLM` expose the current
recommended model. Legacy v1 and explicit v2 classes remain available
for compatibility.
"""

# Default PULSE model (current generation)
from .pulse_model import PulseConfig, PulseForCausalLM, PulseModel

# Explicit v2 implementation
from .pulse_v2 import (
    PulseV2Config,
    PulseV2ForCausalLM,
    PulseV2,
)

# Legacy v1 implementation
from .pulse_legacy import (
    PulseConfig as LegacyPulseConfig,
    PulseForCausalLM as LegacyPulseForCausalLM,
    PulseModel as LegacyPulseModel,
)

__all__ = [
    # Default model
    "PulseConfig",
    "PulseForCausalLM",
    "PulseModel",
    # Explicit v2
    "PulseV2Config",
    "PulseV2ForCausalLM",
    "PulseV2",
    # Legacy v1
    "LegacyPulseConfig",
    "LegacyPulseForCausalLM",
    "LegacyPulseModel",
]
