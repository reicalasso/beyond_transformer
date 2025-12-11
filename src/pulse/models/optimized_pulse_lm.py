"""
Optimized PULSE Language Model.

This module provides a highly optimized version of PULSE for production use.
Key optimizations:
- Fused attention operations
- Memory-efficient state management
- Optimized state propagation
- SwiGLU FFN
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.optimized_attention import (
    OptimizedPulseLayer,
    OptimizedSelfAttention,
    OptimizedStateAttention,
    OptimizedStatePropagator,
)


@dataclass
class OptimizedPulseConfig:
    """Configuration for optimized PULSE model."""
    
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_states: int = 32
    state_dim: int = 768
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.0  # No dropout for inference
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizedPulseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class OptimizedPulseEmbeddings(nn.Module):
    """Optimized embeddings with fused operations."""

    def __init__(self, config: OptimizedPulseConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Pre-compute position indices
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        # Fused embedding lookup
        embeddings = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.norm(embeddings)


class OptimizedPulseModel(nn.Module):
    """Optimized PULSE base model."""

    def __init__(self, config: OptimizedPulseConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = OptimizedPulseEmbeddings(config)

        # Initial states
        self.initial_states = nn.Parameter(
            torch.randn(1, config.num_states, config.state_dim) * config.initializer_range
        )

        # Optimized layers
        self.layers = nn.ModuleList([
            OptimizedPulseLayer(
                hidden_size=config.hidden_size,
                state_dim=config.state_dim,
                num_heads=config.num_heads,
                num_states=config.num_states,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask
            past_key_values: Optional KV cache
            use_cache: Whether to return KV cache

        Returns:
            hidden_states, final_states, new_past_key_values
        """
        batch_size = input_ids.shape[0]

        # Embeddings
        hidden_states = self.embeddings(input_ids)

        # Initialize states
        states = self.initial_states.expand(batch_size, -1, -1)

        # Process through layers
        new_past_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, states, new_kv = layer(hidden_states, states, past_kv)

            if use_cache:
                new_past_key_values.append(new_kv)

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        return hidden_states, states, new_past_key_values


class OptimizedPulseForCausalLM(nn.Module):
    """Optimized PULSE for causal language modeling."""

    def __init__(self, config: OptimizedPulseConfig) -> None:
        super().__init__()
        self.config = config
        self.model = OptimizedPulseModel(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask
            labels: Optional labels for loss computation
            past_key_values: Optional KV cache
            use_cache: Whether to return KV cache

        Returns:
            Dictionary with logits, loss (if labels provided), past_key_values
        """
        hidden_states, states, new_past_key_values = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        output = {"logits": logits, "states": states}

        if use_cache:
            output["past_key_values"] = new_past_key_values

        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Optimized generation with KV caching.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initial forward pass
        outputs = self(input_ids, use_cache=True)
        past_key_values = outputs["past_key_values"]
        next_token_logits = outputs["logits"][:, -1, :]

        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Forward with cache
            outputs = self(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs["past_key_values"]
            next_token_logits = outputs["logits"][:, -1, :]

        return generated
