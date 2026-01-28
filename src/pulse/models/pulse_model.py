"""
Primary PULSE language model interface.

This module exposes configuration and model classes that wrap the
latest PULSE architecture, while keeping legacy v1/v2 implementations
available under separate names.
"""

from .pulse_v2 import PulseV2Config, PulseV2, PulseV2ForCausalLM

# Public, default configuration and model types
PulseConfig = PulseV2Config


class PulseModel(PulseV2):
    """Default PULSE base model."""

    def __init__(self, config: PulseConfig):
        super().__init__(config)


class PulseForCausalLM(PulseV2ForCausalLM):
    """Default PULSE causal language model."""

    def __init__(self, config: PulseConfig):
        super().__init__(config)

