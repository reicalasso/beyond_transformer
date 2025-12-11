"""
PULSE Language Models.

This module implements language modeling capabilities for PULSE:
- PulseForCausalLM: Autoregressive language model
- PULSEForMaskedLM: Masked language model (BERT-style)
- PulseForSequenceClassification: Sequence classification
- PulseForTokenClassification: Token-level classification (NER, POS)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.adaptive_state import AdaptiveStateAllocator, HierarchicalStateManager
from ..core.attention import CausalStateAttention, LinearAttention, SparseStateAttention
from ..modules.state_propagator import StatePropagator


@dataclass
class PulseConfig:
    """Configuration for PULSE language models."""
    
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_states: int = 32
    state_dim: int = 768
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_adaptive_states: bool = True
    use_hierarchical_states: bool = False
    gate_type: str = "gru"
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PulseConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class PulseEmbeddings(nn.Module):
    """Embeddings for PULSE models."""

    def __init__(self, config: PulseConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        # Initialize position ids
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            position_ids: Optional position IDs [batch_size, seq_len]

        Returns:
            Embeddings tensor [batch_size, seq_len, hidden_size]
        """
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = word_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class PulseLayer(nn.Module):
    """Single PULSE layer with state-based processing."""

    def __init__(self, config: PulseConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # State propagator
        self.state_propagator = StatePropagator(
            state_dim=config.state_dim,
            gate_type=config.gate_type,
            num_heads=config.num_heads,
            enable_communication=True,
        )

        # Attention mechanisms
        self.self_attention = CausalStateAttention(
            dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            max_seq_len=config.max_position_embeddings,
        )

        self.state_attention = SparseStateAttention(
            token_dim=config.hidden_size,
            state_dim=config.state_dim,
            num_heads=config.num_heads,
            top_k=min(8, config.num_states),
            dropout=config.attention_dropout,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )

        # Layer norms
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.state_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # State projection
        self.state_proj = nn.Linear(config.state_dim, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for PULSE layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            states: State tensor [batch_size, num_states, state_dim]
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache for incremental decoding

        Returns:
            Tuple of (output, updated_states, new_kv_cache)
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, new_kv_cache = self.self_attention(hidden_states, kv_cache)
        hidden_states = residual + hidden_states

        # State attention with residual
        residual = hidden_states
        hidden_states = self.state_norm(hidden_states)
        state_output, _ = self.state_attention(hidden_states, states, attention_mask)
        hidden_states = residual + state_output

        # Update states based on hidden states
        # Use mean pooling of hidden states as input to state propagator
        state_input = hidden_states.mean(dim=1, keepdim=True).expand(-1, states.shape[1], -1)
        state_input = self.state_proj(state_input) if state_input.shape[-1] != self.config.state_dim else state_input
        
        # Project hidden states to state dimension if needed
        if hidden_states.shape[-1] != self.config.state_dim:
            state_input = F.linear(state_input, self.state_proj.weight.t())
        
        updated_states = self.state_propagator(states, state_input)

        # FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states, updated_states, new_kv_cache


class PulseModel(nn.Module):
    """Base PULSE model without task-specific head."""

    def __init__(self, config: PulseConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = PulseEmbeddings(config)

        # State initialization
        if config.use_adaptive_states:
            self.state_allocator = AdaptiveStateAllocator(
                input_dim=config.hidden_size,
                state_dim=config.state_dim,
                min_states=4,
                max_states=config.num_states,
                initial_states=config.num_states // 2,
            )
        else:
            self.initial_states = nn.Parameter(
                torch.randn(1, config.num_states, config.state_dim) * config.initializer_range
            )

        # Hierarchical state manager (optional)
        if config.use_hierarchical_states:
            self.hierarchical_states = HierarchicalStateManager(
                state_dim=config.state_dim,
                num_token_states=config.num_states,
                num_chunk_states=config.num_states // 2,
                num_global_states=config.num_states // 4,
            )

        # PULSE layers
        self.layers = nn.ModuleList([
            PulseLayer(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_initial_states(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get initial states for a batch."""
        if self.config.use_adaptive_states:
            # Will be computed dynamically
            return None
        else:
            return self.initial_states.expand(batch_size, -1, -1).to(device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for PULSE model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: Cached key-value pairs for incremental decoding
            use_cache: Whether to return key-value cache
            output_states: Whether to output intermediate states

        Returns:
            Dictionary with 'last_hidden_state', 'states', and optionally 'past_key_values'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        hidden_states = self.embeddings(input_ids, position_ids)

        # Initialize states
        if self.config.use_adaptive_states:
            states, state_mask = self.state_allocator(hidden_states)
        else:
            states = self.get_initial_states(batch_size, device)

        # Process through layers
        all_states = [] if output_states else None
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, states, new_kv = layer(
                hidden_states,
                states,
                attention_mask,
                past_kv,
            )

            if use_cache:
                present_key_values.append(new_kv)

            if output_states:
                all_states.append(states.clone())

        # Apply hierarchical processing if enabled
        if self.config.use_hierarchical_states:
            hidden_states, _ = self.hierarchical_states(hidden_states)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        output = {
            "last_hidden_state": hidden_states,
            "states": states,
        }

        if use_cache:
            output["past_key_values"] = present_key_values

        if output_states:
            output["all_states"] = all_states

        return output


class PulseForCausalLM(nn.Module):
    """PULSE model for causal language modeling (autoregressive generation)."""

    def __init__(self, config: PulseConfig) -> None:
        super().__init__()
        self.config = config
        self.model = PulseModel(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
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
        Forward pass for causal LM.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            use_cache: Whether to return cache

        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), and optionally 'past_key_values'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs["last_hidden_state"]
        logits = self.lm_head(hidden_states)

        output = {"logits": logits}

        if labels is not None:
            # Shift for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        if use_cache:
            output["past_key_values"] = outputs["past_key_values"]

        output["states"] = outputs["states"]

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
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample or use greedy decoding
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID

        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        eos_token_id = eos_token_id or self.config.eos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Track which sequences are done
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)

        past_key_values = None
        generated = input_ids

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Update generated sequence
            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)

            if not unfinished.any():
                break

        return generated


class PulseForSequenceClassification(nn.Module):
    """PULSE model for sequence classification."""

    def __init__(self, config: PulseConfig, num_labels: int = 2) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = PulseModel(config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token or mean pooling
        hidden_states = outputs["last_hidden_state"]

        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.classifier(pooled)

        output = {"logits": logits}

        if labels is not None:
            if self.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
            output["loss"] = loss

        return output


class PulseForTokenClassification(nn.Module):
    """PULSE model for token classification (NER, POS tagging)."""

    def __init__(self, config: PulseConfig, num_labels: int = 9) -> None:
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.model = PulseModel(config)

        # Token classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for token classification.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs["last_hidden_state"]
        logits = self.classifier(hidden_states)

        output = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output
