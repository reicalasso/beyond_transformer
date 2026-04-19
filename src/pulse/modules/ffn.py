"""Feed-forward networks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU FFN: ``down(silu(gate(x)) * up(x))``.

    Args:
        dim: Input/output dimension.
        intermediate: Hidden width. Typically a multiple of 64 for tensor cores.
        bias: Whether linear layers carry biases (default False).
    """

    def __init__(self, dim: int, intermediate: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(dim, intermediate, bias=bias)
        self.up = nn.Linear(dim, intermediate, bias=bias)
        self.down = nn.Linear(intermediate, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
