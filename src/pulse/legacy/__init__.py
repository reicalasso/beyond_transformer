"""
PULSE v3 (legacy) — preserved for reproducibility.

This is the original dual-timescale linear-attention prototype. It has been
superseded by the modern PULSE architecture in :mod:`pulse` (gated delta-rule
recurrence with matrix-valued state, hybrid sliding-window attention, RoPE,
QK-norm, logit soft-cap, μP-style init).

Use this module only when you explicitly need the v3 prototype:

    from pulse.legacy import PulseConfig, PulseForCausalLM
"""

from .model import PulseConfig, PulseModel, PulseForCausalLM

__all__ = ["PulseConfig", "PulseModel", "PulseForCausalLM"]