"""
PULSE State Space Model (SSM) Block

Provides Mamba-style selective state space model:
- O(n) complexity for sequence processing
- Selective mechanism for input-dependent dynamics
- Hardware-efficient implementation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Mamba style.
    
    Key features:
    - Input-dependent state transitions (selective)
    - O(n) complexity vs O(n²) for attention
    - Efficient parallel scan implementation
    
    Args:
        hidden_size: Model dimension
        state_size: SSM state dimension (default: 16)
        dt_rank: Rank for delta projection (default: hidden_size // 16)
        expand: Expansion factor for inner dimension
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        dt_rank: int = None,
        expand: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.dt_rank = dt_rank or max(hidden_size // 16, 1)
        self.expand = expand
        self.inner_size = hidden_size * expand
        
        # Input projection
        self.in_proj = nn.Linear(hidden_size, self.inner_size * 2, bias=False)
        
        # SSM parameters
        # A is structured as diagonal for efficiency
        A = torch.arange(1, state_size + 1, dtype=torch.float32).repeat(self.inner_size, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.inner_size))
        
        # Selective projections (input-dependent B, C, delta)
        self.x_proj = nn.Linear(self.inner_size, self.dt_rank + state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_size, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.inner_size, hidden_size, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.inner_size, self.inner_size,
            kernel_size=4, padding=3, groups=self.inner_size
        )
        
        # Initialize dt bias for stability
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.bias, -dt_init_std, dt_init_std)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            state: Optional initial state [batch, inner_size, state_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            final_state: [batch, inner_size, state_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_conv = x_inner.transpose(1, 2)  # [batch, inner, seq]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = x_conv.transpose(1, 2)  # [batch, seq, inner]
        x_inner = F.silu(x_conv)
        
        # Selective projections
        x_dbl = self.x_proj(x_inner)
        dt, B, C = x_dbl.split([self.dt_rank, self.state_size, self.state_size], dim=-1)
        
        # Delta (discretization step)
        dt = F.softplus(self.dt_proj(dt))  # [batch, seq, inner]
        
        # A matrix (diagonal, negative for stability)
        A = -torch.exp(self.A_log)  # [inner, state]
        
        # Run SSM
        output, final_state = self._ssm_scan(x_inner, dt, A, B, C, state)
        
        # Add skip connection with D
        output = output + x_inner * self.D
        
        # Gate and project
        output = output * F.silu(z)
        output = self.out_proj(output)
        
        return output, final_state
    
    def _ssm_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential SSM scan.
        
        For production, this should use parallel scan for efficiency.
        """
        batch_size, seq_len, inner_size = x.shape
        state_size = self.state_size
        
        # Initialize state
        if state is None:
            state = torch.zeros(batch_size, inner_size, state_size, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # Get current inputs
            x_t = x[:, t, :]  # [batch, inner]
            dt_t = dt[:, t, :]  # [batch, inner]
            B_t = B[:, t, :]  # [batch, state]
            C_t = C[:, t, :]  # [batch, state]
            
            # Discretize A and B
            # dA = exp(dt * A)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [batch, inner, state]
            # dB = dt * B (simplified)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [batch, inner, state]
            
            # State update: h = dA * h + dB * x
            state = dA * state + dB * x_t.unsqueeze(-1)
            
            # Output: y = C * h
            y_t = (state * C_t.unsqueeze(1)).sum(dim=-1)  # [batch, inner]
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # [batch, seq, inner]
        return output, state


class SSMBlock(nn.Module):
    """
    Complete SSM block with normalization and residual.
    
    Args:
        hidden_size: Model dimension
        state_size: SSM state dimension
        expand: Expansion factor
        norm_eps: LayerNorm epsilon
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        expand: int = 2,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.ssm = SelectiveSSM(hidden_size, state_size, expand=expand)
        self.norm = nn.LayerNorm(hidden_size, eps=norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with residual connection."""
        residual = x
        x = self.norm(x)
        x, state = self.ssm(x, state)
        return residual + x, state
