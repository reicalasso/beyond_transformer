"""
PULSE Normalization Layers

Provides optimized normalization implementations:
- RMSNorm: Faster than LayerNorm, same quality
- LayerNorm: Standard implementation for compatibility
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Faster than LayerNorm (no mean computation), empirically same quality.
    Used in Llama, Mistral, and other modern architectures.
    
    Args:
        hidden_size: Dimension to normalize
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
    
    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
