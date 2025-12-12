"""
PULSE Feed-Forward Networks

Provides various FFN implementations:
- SwiGLU: Gated Linear Unit with SiLU activation (Llama style)
- GeGLU: Gated Linear Unit with GELU activation
- MLP: Standard MLP with configurable activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    Uses gated linear unit with SiLU (Swish) activation.
    Better than standard MLP empirically.
    
    Used in Llama, Mistral, and other modern architectures.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (typically 2.7x hidden_size)
        bias: Whether to use bias in linear layers
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


class GeGLU(nn.Module):
    """
    GeGLU Feed-Forward Network.
    
    Uses gated linear unit with GELU activation.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension
        bias: Whether to use bias
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
        return self.down_proj(F.gelu(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    """
    Standard MLP with configurable activation.
    
    Args:
        hidden_size: Input/output dimension
        intermediate_size: Hidden dimension (typically 4x hidden_size)
        activation: Activation function ('gelu', 'relu', 'silu')
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 4
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        
        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
