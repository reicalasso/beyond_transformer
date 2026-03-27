"""
PULSE v3 — model definitions.

PulseConfig   : flat dataclass, serializable to/from dict.
PulseModel    : bare encoder (returns hidden states + states).
PulseForCausalLM : adds LM head, loss, and O(n) incremental generate().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import RMSNorm, DualStateBlock


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PulseConfig:
    # Vocabulary
    vocab_size: int = 50257
    pad_token_id: Optional[int] = None

    # Architecture
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    ffn_mult: float = 2.7
    kernel_size: int = 4

    # Timescales (initial values; both are learnable per-head)
    fast_decay: float = 0.70
    slow_decay: float = 0.97

    # Sequence
    max_seq_len: int = 8192

    # Regularisation
    dropout: float = 0.0
    norm_eps: float = 1e-6

    # Misc
    tie_embeddings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PulseConfig:
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class PulseModel(nn.Module):
    """
    Stack of DualStateBlocks.  Returns hidden states and per-layer states.

    No positional encoding — fast/slow decay provides implicit position bias.
    """

    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList([
            DualStateBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                ffn_mult=config.ffn_mult,
                kernel_size=config.kernel_size,
                fast_decay=config.fast_decay,
                slow_decay=config.slow_decay,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,                    # [B, T]
        layer_states: Optional[List] = None,        # list of per-layer state tuples
        mask: Optional[torch.Tensor] = None,        # [B, T] 1=valid, 0=pad
    ) -> Tuple[torch.Tensor, List]:
        B, T = input_ids.shape

        x = self.drop(self.embed(input_ids))        # [B, T, D]

        # Build [B, 1, 1, T] attention mask for inner modules
        if mask is not None:
            attn_mask = mask.view(B, 1, 1, T).to(x.dtype)
        else:
            attn_mask = None

        if layer_states is None:
            layer_states = [None] * len(self.layers)

        new_states: List = []
        for layer, state in zip(self.layers, layer_states):
            x, new_state = layer(x, state=state, mask=attn_mask)
            new_states.append(new_state)

        return self.norm(x), new_states


# ---------------------------------------------------------------------------
# Causal LM wrapper
# ---------------------------------------------------------------------------

class PulseForCausalLM(nn.Module):
    """
    PULSE v3 causal language model.

    forward()  — training / teacher-forced evaluation.
    generate() — O(n) incremental autoregressive decoding.
    """

    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.model  = PulseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,                    # [B, T]
        labels: Optional[torch.Tensor] = None,      # [B, T]  -100 = ignore
        layer_states: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> Dict[str, torch.Tensor]:
        hidden, new_states = self.model(input_ids, layer_states, attention_mask)
        logits = self.lm_head(hidden)               # [B, T, V]

        out: Dict[str, Any] = {
            "logits": logits,
            "layer_states": new_states,
        }

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            out["loss"] = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return out

    # ------------------------------------------------------------------
    # Incremental generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,            # [B, prompt_len]
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        O(n) generation — prompt encoded once, then single-token steps.

        Layer states (attn KV + conv buffer) carry all prior context so
        each decode step runs in O(1) w.r.t. sequence length.
        """
        generated = input_ids.clone()

        # --- Prompt pass ---
        pad_id = self.config.pad_token_id
        prompt_mask = (generated != pad_id).long() if pad_id is not None else None

        out = self(generated, attention_mask=prompt_mask)
        layer_states = out["layer_states"]

        next_token = self._sample(
            out["logits"][:, -1], temperature, top_k, top_p,
            repetition_penalty, generated,
        )
        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return generated

        # --- Incremental decode ---
        for _ in range(max_new_tokens - 1):
            out = self(next_token, layer_states=layer_states)
            layer_states = out["layer_states"]

            next_token = self._sample(
                out["logits"][:, -1], temperature, top_k, top_p,
                repetition_penalty, generated,
            )
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(
        self,
        logits: torch.Tensor,               # [B, V]
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
        generated: torch.Tensor,            # [B, T_so_far]
    ) -> torch.Tensor:
        logits = logits.float() / max(temperature, 1e-8)

        # Repetition penalty
        if rep_penalty != 1.0:
            for i in range(logits.size(0)):
                unique = generated[i].unique()
                logits[i, unique] = torch.where(
                    logits[i, unique] > 0,
                    logits[i, unique] / rep_penalty,
                    logits[i, unique] * rep_penalty,
                )

        # Top-k
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits[logits < kth] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0]  = False
            logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)