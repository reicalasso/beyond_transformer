"""Composite blocks: Delta-recurrence block and SWA-attention block.

Both follow a Pre-Norm sandwich:

.. code-block:: text

    x → RMSNorm → ShortCausalConv1d → mixer (Delta or SWA) → residual
      → RMSNorm → SwiGLU FFN → residual

The short conv before the mixer is the Mamba-style "local mixer": it gives
the recurrence/attention richer per-channel features (similar effect to a
positional bias) and is fully streaming.

Each block carries a ``BlockState`` tuple of ``(conv_state, mixer_state)``
so the same interface works in both prefill and incremental decode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from .conv import ShortCausalConv1d
from .delta import DeltaState, GatedDeltaRule
from .ffn import SwiGLU
from .norm import RMSNorm
from .swa import SlidingWindowAttention, SWACache

MixerState = Union[DeltaState, SWACache]


@dataclass
class BlockState:
    conv_state: torch.Tensor | None
    mixer_state: MixerState | None

    def detach(self) -> BlockState:
        cs = self.conv_state.detach() if self.conv_state is not None else None
        ms: MixerState | None
        if self.mixer_state is None:
            ms = None
        elif isinstance(self.mixer_state, DeltaState):
            ms = self.mixer_state.detach()
        else:
            ms = SWACache(
                k=self.mixer_state.k.detach(),
                v=self.mixer_state.v.detach(),
                offset=self.mixer_state.offset,
            )
        return BlockState(conv_state=cs, mixer_state=ms)


class _BaseBlock(nn.Module):
    """Shared norm + conv + FFN scaffolding."""

    def __init__(
        self,
        hidden_size: int,
        ffn_mult: float,
        conv_kernel_size: int,
        norm_eps: float,
    ):
        super().__init__()
        intermediate = int(hidden_size * ffn_mult)
        self.norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.conv = ShortCausalConv1d(hidden_size, kernel_size=conv_kernel_size, activation="silu")
        self.ffn = SwiGLU(hidden_size, intermediate)

    def _conv_then_residual(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        xn = self.norm1(x)
        xc, new_conv = self.conv(xn, state=conv_state)
        return residual, xc, new_conv

    def _ffn_residual(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm2(x))


class DeltaBlock(_BaseBlock):
    """Recurrent block: short conv → gated delta-rule → SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_mult: float = 2.7,
        conv_kernel_size: int = 4,
        chunk_size: int = 64,
        qk_norm: bool = True,
        gate_bias_init: float = 4.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__(hidden_size, ffn_mult, conv_kernel_size, norm_eps)
        self.mixer = GatedDeltaRule(
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_norm=qk_norm,
            chunk_size=chunk_size,
            gate_bias_init=gate_bias_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: BlockState | None = None,
        attention_mask: torch.Tensor | None = None,  # unused here, accepted for symmetry
    ) -> tuple[torch.Tensor, BlockState]:
        del attention_mask  # delta path uses no mask; padding handled at trainer level
        conv_state = state.conv_state if state is not None else None
        mixer_state = state.mixer_state if state is not None else None
        if mixer_state is not None and not isinstance(mixer_state, DeltaState):
            raise TypeError("DeltaBlock requires DeltaState, got SWACache")

        residual, xc, new_conv = self._conv_then_residual(x, conv_state)
        mixer_out, new_mixer = self.mixer(xc, state=mixer_state)
        x = residual + mixer_out
        x = self._ffn_residual(x)
        return x, BlockState(conv_state=new_conv, mixer_state=new_mixer)


class AttentionBlock(_BaseBlock):
    """Recall block: short conv → sliding-window softmax attention → SwiGLU FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int,
        num_kv_heads: int | None = None,
        ffn_mult: float = 2.7,
        conv_kernel_size: int = 4,
        qk_norm: bool = True,
        rope_base: float = 10_000.0,
        rope_max_seq_len: int = 4096,
        norm_eps: float = 1e-6,
    ):
        super().__init__(hidden_size, ffn_mult, conv_kernel_size, norm_eps)
        self.mixer = SlidingWindowAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            window_size=window_size,
            num_kv_heads=num_kv_heads,
            qk_norm=qk_norm,
            rope_base=rope_base,
            rope_max_seq_len=rope_max_seq_len,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: BlockState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, BlockState]:
        conv_state = state.conv_state if state is not None else None
        mixer_state = state.mixer_state if state is not None else None
        if mixer_state is not None and not isinstance(mixer_state, SWACache):
            raise TypeError("AttentionBlock requires SWACache, got DeltaState")

        residual, xc, new_conv = self._conv_then_residual(x, conv_state)
        mixer_out, new_mixer = self.mixer(xc, cache=mixer_state, attention_mask=attention_mask)
        x = residual + mixer_out
        x = self._ffn_residual(x)
        return x, BlockState(conv_state=new_conv, mixer_state=new_mixer)
