"""Gated Delta Rule with matrix-valued state.

This is the centerpiece of the modern PULSE recurrence. Per token and per head,
the state ``S_t ∈ R^{Dh × Dh}`` evolves as:

.. code-block:: text

    S_t = α_t · S_{t-1} · (I - β_t k_t k_t^T) + β_t v_t k_t^T
    o_t = S_t @ q_t

where:
    α_t ∈ (0, 1)  is a data-dependent forget gate (per-token, per-head scalar).
    β_t ∈ (0, 1]  is a data-dependent write strength (per-token, per-head scalar).
    q, k, v       are projections of the input, optionally L2-normalized
                  (QK-norm) for numerical stability.

This generalises both:
  * GLA / linear attention  (β_t = 1, no delta correction)
  * RWKV-7 "Goose"          (data-dependent decay, scalar β)
  * Mamba-2 / SSD           (chunkwise structured state evolution)
  * DeltaNet / GatedDeltaNet (Yang et al. 2024)

Two execution paths are exposed:

* :meth:`forward_recurrent` — per-token Python loop, used as the reference
  implementation and as the O(1) incremental-decode path.
* :meth:`forward_chunked`   — chunked iteration with batched intra-chunk
  matmul ops; same FLOPs but fewer Python iterations per token. The future
  Triton kernel will plug into the same interface.

A pytest suite asserts numerical equivalence between the two paths and
between prefill/decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import l2_normalize


@dataclass
class DeltaState:
    """Carry-state for one :class:`GatedDeltaRule` layer.

    Attributes:
        S: Matrix-valued state, shape ``[B, H, Dh, Dh]``.
    """

    S: torch.Tensor

    def detach(self) -> DeltaState:
        return DeltaState(S=self.S.detach())


class GatedDeltaRule(nn.Module):
    """Gated delta-rule recurrence with matrix-valued state.

    Args:
        hidden_size: Model hidden dimension.
        num_heads: Number of attention-style heads. ``head_dim = hidden_size /
            num_heads``.
        qk_norm: Apply L2 normalization to ``q`` and ``k`` per head.
        chunk_size: Chunk length used by :meth:`forward_chunked`. Must divide
            evenly during prefill or a remainder chunk is processed.
        gate_bias_init: Initial bias for the forget-gate sigmoid. Higher values
            push α_t toward 1 at initialization (keep memory longer).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_norm: bool = True,
        chunk_size: int = 64,
        gate_bias_init: float = 4.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.alpha_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=True)

        self.out_gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        nn.init.constant_(self.alpha_proj.bias, gate_bias_init)
        nn.init.zeros_(self.beta_proj.bias)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> DeltaState:
        S = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        return DeltaState(S=S)

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(b, t, h, d).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(b, t, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, d).transpose(1, 2)

        if self.qk_norm:
            q = l2_normalize(q, dim=-1)
            k = l2_normalize(k, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(x)).transpose(1, 2)  # [B, H, T]
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)  # [B, H, T]

        out_gate = F.silu(self.out_gate_proj(x))  # [B, T, hidden]

        return q, k, v, alpha, beta, out_gate

    def forward(
        self,
        x: torch.Tensor,
        state: DeltaState | None = None,
        use_chunked: bool = True,
    ) -> tuple[torch.Tensor, DeltaState]:
        """Forward pass.

        Args:
            x: ``[B, T, hidden_size]``.
            state: Optional carry state from a previous segment.
            use_chunked: If True, use the chunked path during prefill; if
                ``T == 1`` always uses the recurrent path (same result).

        Returns:
            ``(output, new_state)`` with ``output`` shape ``[B, T, hidden_size]``.
        """
        b, t, _ = x.shape
        q, k, v, alpha, beta, out_gate = self._project(x)

        if state is None:
            S0 = torch.zeros(
                b,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=x.device,
                dtype=torch.float32,
            )
        else:
            S0 = state.S.float()

        if t == 1 or not use_chunked:
            o, S_new = self.forward_recurrent(q, k, v, alpha, beta, S0)
        else:
            o, S_new = self.forward_chunked(q, k, v, alpha, beta, S0, self.chunk_size)

        o = o.transpose(1, 2).contiguous().view(b, t, self.hidden_size)  # [B, T, hidden]
        o = o * out_gate
        o = self.out_proj(o)
        return o, DeltaState(S=S_new)

    @staticmethod
    def forward_recurrent(
        q: torch.Tensor,  # [B, H, T, D]
        k: torch.Tensor,  # [B, H, T, D]
        v: torch.Tensor,  # [B, H, T, D]
        alpha: torch.Tensor,  # [B, H, T]
        beta: torch.Tensor,  # [B, H, T]
        S0: torch.Tensor,  # [B, H, D, D]  (fp32)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token recurrent reference.

        Used for incremental decoding (T==1) and as the test oracle.
        Internal accumulation is fp32 regardless of input dtype.
        """
        b, h, t, d = q.shape
        S = S0  # [B, H, D, D] fp32
        outs = []

        qf = q.float()
        kf = k.float()
        vf = v.float()
        af = alpha.float()
        bf = beta.float()

        for i in range(t):
            kt = kf[:, :, i]  # [B, H, D]
            vt = vf[:, :, i]  # [B, H, D]
            qt = qf[:, :, i]  # [B, H, D]
            at = af[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            bt = bf[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            # S_new = α · [S - β (S k) k^T] + β v k^T
            Sk = torch.einsum("bhij,bhj->bhi", S, kt).unsqueeze(-1)  # [B, H, D, 1]
            outer_v = vt.unsqueeze(-1)  # [B, H, D, 1]
            outer_k = kt.unsqueeze(-2)  # [B, H, 1, D]
            S = at * (S - bt * (Sk @ outer_k)) + bt * (outer_v @ outer_k)

            ot = torch.einsum("bhij,bhj->bhi", S, qt)  # [B, H, D]
            outs.append(ot)

        out = torch.stack(outs, dim=2)  # [B, H, T, D]
        return out.to(q.dtype), S

    @staticmethod
    def forward_chunked(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        S0: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked path: identical FLOPs to :meth:`forward_recurrent`.

        Iterates over chunks in Python and over tokens within each chunk
        (the loop body uses batched matmuls). Provides a stable interface
        for a future fused intra-chunk WY-representation kernel.
        """
        b, h, t, d = q.shape
        S = S0
        outs = []

        for start in range(0, t, chunk_size):
            end = min(start + chunk_size, t)
            chunk_out, S = GatedDeltaRule.forward_recurrent(
                q[:, :, start:end],
                k[:, :, start:end],
                v[:, :, start:end],
                alpha[:, :, start:end],
                beta[:, :, start:end],
                S,
            )
            outs.append(chunk_out)

        out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
        return out, S
