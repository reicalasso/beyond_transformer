"""PULSE — modern hybrid recurrent-attention sequence model.

Public API:

* :class:`PulseConfig`
* :class:`PulseModel`
* :class:`PulseForCausalLM`

The legacy v3 dual-timescale prototype is kept under :mod:`pulse.legacy` for
reproducibility.
"""

from .config import PulseConfig
from .model import PulseForCausalLM, PulseModel

__version__ = "4.0.0"

__all__ = ["PulseConfig", "PulseForCausalLM", "PulseModel", "__version__"]
