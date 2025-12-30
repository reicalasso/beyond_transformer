"""
PULSE Feed-Forward Network

SwiGLU: The only FFN you need.
Gated Linear Unit with SiLU activation - proven best empirically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Uses gated linear unit with SiLU (Swish) activation.
    Used in Llama, Mistral, and other modern architectures.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (default: 2.7x hidden_size)
        bias: Whether to use bias (default: False)
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        bias: bool = False,
    ):
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 2.7)
        
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
