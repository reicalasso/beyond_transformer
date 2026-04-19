"""Configuration dataclass for the modern PULSE architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PulseConfig:
    """All hyperparameters for :class:`pulse.PulseForCausalLM`.

    The default layer pattern is a 4:1 hybrid (every 4th block is sliding-window
    attention, others are gated delta-rule). This matches the Samba/Zamba-2
    family "recall every K layers" pattern.
    """

    # Vocabulary
    vocab_size: int = 50_257
    pad_token_id: int | None = None
    tie_embeddings: bool = True

    # Architecture
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    num_kv_heads: int | None = None  # None → equal to num_heads (MHA in SWA blocks)
    ffn_mult: float = 2.7
    conv_kernel_size: int = 4

    # Hybrid layer pattern: index → "delta" | "swa".
    # If ``None``, generated from ``swa_every`` (place SWA at indices that satisfy
    # ``(i + 1) % swa_every == 0``).
    layer_types: list[str] | None = None
    swa_every: int = 4
    swa_window_size: int = 512

    # Recurrence (delta) details
    delta_chunk_size: int = 64
    qk_norm: bool = True
    gate_bias_init: float = 4.0  # logit for sigmoid α; ~0.98 at init

    # RoPE (used by SWA blocks)
    rope_base: float = 10_000.0
    rope_max_seq_len: int = 8192

    # Sequence cap (advisory; does not constrain forward())
    max_seq_len: int = 8192

    # Numerics
    norm_eps: float = 1.0e-6
    logit_soft_cap: float | None = 30.0  # Gemma-style; None to disable
    z_loss_coef: float = 1.0e-4  # Auxiliary log-Z loss (set 0 to disable)

    # Regularization
    dropout: float = 0.0

    # Initialization (μP-style scale on output / projection weights)
    init_std: float = 0.02
    init_scale_residual: bool = True  # scale residual proj std by 1/sqrt(2 * num_layers)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def resolved_layer_types(self) -> list[str]:
        """Return the per-layer type list, generating from ``swa_every`` if needed."""
        if self.layer_types is not None:
            if len(self.layer_types) != self.num_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but "
                    f"num_layers={self.num_layers}"
                )
            for t in self.layer_types:
                if t not in ("delta", "swa"):
                    raise ValueError(f"Unknown layer type: {t}")
            return list(self.layer_types)
        return [
            "swa" if ((i + 1) % self.swa_every == 0) else "delta" for i in range(self.num_layers)
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PulseConfig:
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
