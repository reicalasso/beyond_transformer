"""Short causal depthwise 1D convolution with incremental decode state.

Mamba-style "short conv" applied per-channel before Q/K/V or directly on the
hidden stream. During training we left-pad with zeros; during incremental
decoding we keep a rolling buffer of size ``kernel_size - 1``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ShortCausalConv1d(nn.Module):
    """Depthwise causal 1D convolution with cached state for streaming decode.

    Input/output shape: ``[B, T, C]``.

    Args:
        channels: Number of channels (depthwise: groups == channels).
        kernel_size: Conv kernel size (typical 3 or 4).
        bias: Add a per-channel bias.
        activation: Optional pointwise activation applied after conv.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        bias: bool = True,
        activation: str | None = "silu",
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=0,
            groups=channels,
            bias=bias,
        )
        if activation is None:
            self.act: nn.Module = nn.Identity()
        elif activation == "silu":
            self.act = nn.SiLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the conv with optional carry-in ``state``.

        Args:
            x: ``[B, T, C]``.
            state: ``[B, C, kernel_size - 1]`` carry from a previous step,
                or ``None`` to use causal zero-padding (training/prefill).

        Returns:
            ``(out, new_state)`` where ``out`` has shape ``[B, T, C]`` and
            ``new_state`` has shape ``[B, C, kernel_size - 1]``.
        """
        b, t, c = x.shape
        xc = x.transpose(1, 2)  # [B, C, T]

        pad = self.kernel_size - 1
        if state is None:
            left = xc.new_zeros(b, c, pad)
        else:
            if state.shape != (b, c, pad):
                raise ValueError(
                    f"conv state shape mismatch: got {tuple(state.shape)}, expected {(b, c, pad)}"
                )
            left = state

        xc_padded = torch.cat([left, xc], dim=2)  # [B, C, pad + T]
        new_state = xc_padded[:, :, -pad:].detach() if pad > 0 else xc.new_zeros(b, c, 0)

        y = self.conv(xc_padded)  # [B, C, T]
        y = self.act(y)
        return y.transpose(1, 2).contiguous(), new_state
