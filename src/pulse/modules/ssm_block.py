"""
SSM (State Space Model) Based Layer Implementation

This module implements an SSM-based layer using the Mamba SSM implementation.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import Mamba from the mamba_ssm package
try:
    from mamba_ssm.modules.mamba_simple import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.info("Mamba SSM not available. Using simplified implementation.")


class SSMBlock(nn.Module):
    """
    SSM-based block using Mamba implementation.

    This block implements a State Space Model layer that can be used
    as a component in the PULSE architecture.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: Optional[int] = None,
    ):
        """
        Initialize the SSMBlock.

        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            layer_idx: Layer index (for debugging)
        """
        super(SSMBlock, self).__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.layer_idx = layer_idx

        if MAMBA_AVAILABLE:
            # Use the actual Mamba implementation
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
            )
        else:
            # Fallback implementation if Mamba is not available
            self._create_fallback_implementation()

    def _create_fallback_implementation(self):
        """Create a simplified fallback implementation."""
        self.d_inner = int(self.expand * self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SSMBlock.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        if MAMBA_AVAILABLE:
            return self.mamba(x)
        else:
            return self._fallback_forward(x)

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback forward pass implementation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Project input
        x_and_res = self.in_proj(x)
        x_proj, res = x_and_res.chunk(2, dim=-1)

        # Apply convolution
        x_proj = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_proj)[:, :, : x_proj.size(2)]
        x_conv = x_conv.transpose(1, 2)

        # Apply activation
        x_act = self.act(x_conv)

        # Simple linear transformation as a substitute for SSM
        x_ssm = x_act

        # Apply residual connection
        output = self.out_proj(x_ssm * res)

        return output


# Example usage
if __name__ == "__main__":
    # Test SSMBlock
    batch_size, seq_len, d_model = 2, 10, 64

    # Create SSMBlock
    ssm_block = SSMBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = ssm_block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("SSMBlock test completed successfully!")
