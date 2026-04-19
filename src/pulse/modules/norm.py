"""Normalization primitives.

RMSNorm and an L2-normalize helper used for QK-norm. Both keep activations in
fp32 internally for numerical safety, then cast back to the input dtype.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root mean square layer normalization (no bias, no mean subtraction).

    Args:
        dim: Channel dimension to normalize over (last axis).
        eps: Numerical floor for the inverse RMS.
        elementwise_affine: If True, learn a per-channel gain.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = xf * rms
        if self.weight is not None:
            out = out * self.weight
        return out.to(dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6, dim: int = -1) -> torch.Tensor:
    """L2-normalize along ``dim``. Used for QK-norm in attention layers."""
    dtype = x.dtype
    xf = x.float()
    norm = xf.pow(2).sum(dim=dim, keepdim=True).add(eps).rsqrt()
    return (xf * norm).to(dtype)
