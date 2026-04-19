"""Modern PULSE model definitions.

* :class:`PulseModel`        — bare encoder over the hybrid block stack.
* :class:`PulseForCausalLM`  — adds the LM head, label loss with Z-loss,
  optional logit soft-capping, and an O(1)-per-step incremental
  :meth:`PulseForCausalLM.generate`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PulseConfig
from .modules import (
    AttentionBlock,
    DeltaBlock,
    RMSNorm,
)
from .modules.block import BlockState


class PulseModel(nn.Module):
    """Embedding + hybrid block stack + final norm.

    No global positional encoding: the recurrence has implicit position
    information; SWA layers add RoPE.
    """

    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        self.layer_types = config.resolved_layer_types()
        self.layers = nn.ModuleList()
        for layer_type in self.layer_types:
            if layer_type == "delta":
                self.layers.append(
                    DeltaBlock(
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        ffn_mult=config.ffn_mult,
                        conv_kernel_size=config.conv_kernel_size,
                        chunk_size=config.delta_chunk_size,
                        qk_norm=config.qk_norm,
                        gate_bias_init=config.gate_bias_init,
                        norm_eps=config.norm_eps,
                    )
                )
            else:  # "swa"
                self.layers.append(
                    AttentionBlock(
                        hidden_size=config.hidden_size,
                        num_heads=config.num_heads,
                        window_size=config.swa_window_size,
                        num_kv_heads=config.num_kv_heads,
                        ffn_mult=config.ffn_mult,
                        conv_kernel_size=config.conv_kernel_size,
                        qk_norm=config.qk_norm,
                        rope_base=config.rope_base,
                        rope_max_seq_len=config.rope_max_seq_len,
                        norm_eps=config.norm_eps,
                    )
                )

        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        std = self.config.init_std
        residual_scale = (
            1.0 / math.sqrt(2.0 * max(1, self.config.num_layers))
            if self.config.init_scale_residual
            else 1.0
        )
        residual_proj_names = {"out_proj", "down"}

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

        # Scale down residual-projection layers (post-block out_proj and FFN down).
        for name, p in self.named_parameters():
            short = name.rsplit(".", 1)[-1].replace("weight", "").rstrip(".")
            if any(name.endswith(f".{n}.weight") for n in residual_proj_names):
                with torch.no_grad():
                    p.mul_(residual_scale)
            del short  # silence unused

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def empty_states(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[BlockState]:
        states: list[BlockState] = []
        for layer in self.layers:
            ms: object
            if isinstance(layer, DeltaBlock):
                ms = layer.mixer.empty_state(batch_size, device, torch.float32)
            elif isinstance(layer, AttentionBlock):
                ms = layer.mixer.empty_cache(batch_size, device, dtype)
            else:
                raise TypeError(f"Unknown block type: {type(layer).__name__}")
            cs = torch.zeros(
                batch_size,
                self.config.hidden_size,
                self.config.conv_kernel_size - 1,
                device=device,
                dtype=dtype,
            )
            states.append(BlockState(conv_state=cs, mixer_state=ms))
        return states

    def forward(
        self,
        input_ids: torch.Tensor,
        layer_states: list[BlockState | None] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[BlockState]]:
        x = self.drop(self.embed(input_ids))

        if layer_states is None:
            layer_states = [None] * len(self.layers)
        elif len(layer_states) != len(self.layers):
            raise ValueError(
                f"layer_states has {len(layer_states)} entries but model has "
                f"{len(self.layers)} layers"
            )

        new_states: list[BlockState] = []
        for layer, state in zip(self.layers, layer_states):
            x, new_state = layer(x, state=state, attention_mask=attention_mask)
            new_states.append(new_state)

        return self.norm(x), new_states


class PulseForCausalLM(nn.Module):
    """Causal language model wrapper around :class:`PulseModel`.

    Adds:
      * Optional weight-tied LM head.
      * Logit soft-capping (Gemma-style).
      * Auxiliary z-loss for log-partition stability.
      * O(1)-per-step incremental :meth:`generate`.
    """

    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.model = PulseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _apply_soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.config.logit_soft_cap
        if cap is None:
            return logits
        return torch.tanh(logits / cap) * cap

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        layer_states: list[BlockState | None] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        hidden, new_states = self.model(input_ids, layer_states, attention_mask)
        logits = self.lm_head(hidden)
        logits = self._apply_soft_cap(logits)

        out: dict[str, Any] = {
            "logits": logits,
            "layer_states": new_states,
        }

        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)
            ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

            if self.config.z_loss_coef > 0:
                valid = flat_labels.ne(-100)
                if valid.any():
                    log_z = torch.logsumexp(flat_logits[valid].float(), dim=-1)
                    z_loss = self.config.z_loss_coef * (log_z**2).mean()
                else:
                    z_loss = ce.new_zeros(())
                out["ce_loss"] = ce
                out["z_loss"] = z_loss
                out["loss"] = ce + z_loss
            else:
                out["loss"] = ce

        return out

    # ------------------------------------------------------------------
    # Incremental generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample tokens autoregressively.

        Prefill is done in one shot through the chunked path; subsequent steps
        use the per-token recurrent path with carried state for O(1)-per-step
        compute and memory (modulo the SWA window).
        """
        generated = input_ids.clone()
        device = input_ids.device

        out = self.forward(generated, attention_mask=attention_mask)
        layer_states = out["layer_states"]
        next_tok = self._sample(
            out["logits"][:, -1], temperature, top_k, top_p, repetition_penalty, generated
        )
        generated = torch.cat([generated, next_tok], dim=1)
        if eos_token_id is not None and (next_tok == eos_token_id).all():
            return generated

        for _ in range(max_new_tokens - 1):
            out = self.forward(next_tok, layer_states=layer_states)
            layer_states = out["layer_states"]
            next_tok = self._sample(
                out["logits"][:, -1], temperature, top_k, top_p, repetition_penalty, generated
            )
            generated = torch.cat([generated, next_tok], dim=1)
            if eos_token_id is not None and (next_tok == eos_token_id).all():
                break

        del device
        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        rep_penalty: float,
        history: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float() / max(temperature, 1e-8)

        if rep_penalty != 1.0:
            for i in range(logits.size(0)):
                unique = history[i].unique()
                vals = logits[i, unique]
                logits[i, unique] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)

        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, sorted_idx, remove)
            logits = logits.masked_fill(mask, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
