"""
PULSE v2 - Minimal Language Model

Radically simplified architecture:
- UnifiedBlock: Single primitive for local+global processing
- RecurrentState: Single compressed state vector
- SimpleMemory: Optional LRU cache

No SSM/MoE/MoD complexity. Just what works.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.norm import RMSNorm
from ..core.unified import UnifiedBlock, RecurrentState
from ..core.memory import KeyValueMemory


@dataclass
class PulseV2Config:
    """PULSE v2 configuration - minimal and clean."""
    
    # Core dimensions
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 8
    intermediate_size: int = None  # Defaults to hidden_size * 2.7
    
    # Sequence
    max_seq_len: int = 8192
    
    # UnifiedBlock settings
    kernel_size: int = 4
    decay: float = 0.95
    
    # Optional features
    use_memory: bool = False
    memory_capacity: int = 64
    use_recurrent_state: bool = True
    
    # Regularization
    dropout: float = 0.0
    
    # Other
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    pad_token_id: int = None  # Set from tokenizer; None means no padding_idx in Embedding
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 2.7)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a plain dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PulseV2Config":
        """Construct configuration from a dictionary, ignoring unknown keys."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PulseV2(nn.Module):
    """
    PULSE v2 base model.
    
    Simple stack of UnifiedBlocks with optional recurrent state.
    """
    
    def __init__(self, config: PulseV2Config):
        super().__init__()
        self.config = config
        
        # Token embeddings (no position - handled by local conv + linear attn decay)
        self.embed = nn.Embedding(
            config.vocab_size, config.hidden_size,
            padding_idx=config.pad_token_id if config.pad_token_id is not None else None,
        )
        
        # Unified layers
        self.layers = nn.ModuleList([
            UnifiedBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                kernel_size=config.kernel_size,
                decay=config.decay,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ])
        
        # Optional recurrent state.
        # The state is projected and added to the initial embedding so that
        # cross-chunk context actually influences computation (read path).
        # It is then updated from the final hidden states after all layers
        # (write path). Per-layer injection was avoided to prevent collapse.
        if config.use_recurrent_state:
            self.recurrent = RecurrentState(config.hidden_size)
            self.recurrent_in_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        else:
            self.recurrent = None
            self.recurrent_in_proj = None
        
        # Optional external key–value memory with learnable residual gate
        if config.use_memory:
            self.memory = KeyValueMemory(config.hidden_size, config.memory_capacity)
            self.memory_gate = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        else:
            self.memory = None
            self.memory_gate = None
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        recurrent_state: torch.Tensor = None,
        layer_states: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        update_recurrent: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            recurrent_state: Optional [batch, hidden_size]
            layer_states: Optional list of per-layer (attn_state, conv_state) tuples
            attention_mask: Optional [batch, seq_len] with 1=token, 0=pad
            update_recurrent: Whether to update the recurrent state.  Set to
                False during single-token decode to match the training regime
                (one update per full chunk, not per token).
            
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            new_recurrent_state: Updated recurrent state (or unchanged if update_recurrent=False)
            new_layer_states: List of per-layer (attn_state, conv_state) tuples
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        # Build padding mask if provided: [batch, seq_len] -> [batch, 1, 1, seq_len]
        if attention_mask is not None:
            pad_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        else:
            pad_mask = None
        
        # Initialize recurrent state
        if self.recurrent is not None and recurrent_state is None:
            recurrent_state = self.recurrent.get_initial_state(batch_size)
        
        # Condition the initial embedding on cross-chunk context (read path).
        if self.recurrent_in_proj is not None and recurrent_state is not None:
            x = x + self.recurrent_in_proj(recurrent_state).unsqueeze(1)
        
        # Process layers
        new_layer_states: List = []
        if layer_states is None:
            layer_states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, lstate = layer(x, state=layer_states[i], attention_mask=pad_mask)
            new_layer_states.append(lstate)
        
        # Update recurrent state once after all layers.
        # Skipped during single-token decode to match training regime.
        if self.recurrent is not None and update_recurrent:
            recurrent_state = self.recurrent(x, recurrent_state)
        
        # Memory augmentation (optional) — read, gate, then write
        if self.memory is not None:
            mem_output = self.memory.read_attend(x)
            combined = torch.cat([x, mem_output], dim=-1)
            g = torch.sigmoid(self.memory_gate(combined))
            x = g * x + (1 - g) * mem_output
            # Write last-token summary so future reads have content
            self.memory.write(x[:, -1, :].detach())
        
        # Final norm
        x = self.norm(x)
        
        return x, recurrent_state, new_layer_states


class PulseV2ForCausalLM(nn.Module):
    """PULSE v2 for causal language modeling."""
    
    def __init__(self, config: PulseV2Config):
        super().__init__()
        self.config = config
        self.model = PulseV2(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_embeddings:
            self.lm_head.weight = self.model.embed.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        recurrent_state: torch.Tensor = None,
        layer_states: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            labels: Optional [batch, seq_len] for training
            recurrent_state: Optional recurrent state
            layer_states: Optional list of per-layer (attn_state, conv_state) tuples
            
        Returns:
            Dict with logits, loss (if labels), recurrent_state, layer_states
        """
        hidden, recurrent_state, layer_states = self.model(
            input_ids,
            recurrent_state=recurrent_state,
            layer_states=layer_states,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden)
        
        output = {
            "logits": logits,
            "recurrent_state": recurrent_state,
            "layer_states": layer_states,
        }
        
        if labels is not None:
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
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Incremental autoregressive generation.

        Processes the prompt in a single forward pass to build the recurrent
        state and per-layer linear-attention states, then decodes one token
        at a time by feeding only the most recently generated token.  This
        gives O(n) generation cost instead of the O(n²) cost of re-running
        the full sequence every step.
        
        Args:
            input_ids: [batch, seq_len] prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 to disable)
            top_p: Nucleus sampling threshold (1.0 to disable)
            eos_token_id: Stop token (generation halts when all batch items emit this)
            repetition_penalty: Penalize tokens that already appear in the
                generated sequence (1.0 = no penalty, >1.0 = less repetition).
            
        Returns:
            Generated token IDs [batch, prompt_len + num_generated]
        """
        generated = input_ids.clone()

        # ── Prompt encoding pass ──────────────────────────────────────────────────────────
        # Run the full prompt once to build recurrent + per-layer states.
        # Recurrent state IS updated here (full chunk, matches training).
        if self.config.pad_token_id is not None:
            prompt_mask = (generated != self.config.pad_token_id).long()
        else:
            prompt_mask = None
        outputs = self(
            generated,
            recurrent_state=None,
            layer_states=None,
            attention_mask=prompt_mask,
        )
        recurrent_state = outputs["recurrent_state"]
        layer_states = outputs.get("layer_states")

        # Sample the first new token from the last prompt position.
        next_token = self._sample(
            outputs["logits"][:, -1, :], temperature, top_k, top_p,
            repetition_penalty=repetition_penalty, generated_ids=generated,
        )
        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            return generated

        # ── Incremental decoding ──────────────────────────────────────────────────────────
        # During single-token steps the recurrent state is FROZEN to avoid
        # the train-test mismatch (trained: one update per full chunk;
        # inference would otherwise update per token).
        for _ in range(max_new_tokens - 1):
            hidden, _, layer_states = self.model(
                next_token,
                recurrent_state=recurrent_state,
                layer_states=layer_states,
                attention_mask=None,
                update_recurrent=False,  # freeze recurrent state
            )
            logits = self.lm_head(hidden)

            next_token = self._sample(
                logits[:, -1, :], temperature, top_k, top_p,
                repetition_penalty=repetition_penalty, generated_ids=generated,
            )
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float = 1.0,
        generated_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample one token per batch item from logits."""
        # Repetition penalty: reduce probability of already-generated tokens.
        if repetition_penalty != 1.0 and generated_ids is not None:
            for i in range(logits.size(0)):
                seen = generated_ids[i].unique()
                pos = logits[i, seen] > 0
                logits[i, seen] = torch.where(
                    pos,
                    logits[i, seen] / repetition_penalty,
                    logits[i, seen] * repetition_penalty,
                )

        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float('-inf')

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            logits[remove.scatter(1, sorted_idx, remove)] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def write_to_memory(self, key: torch.Tensor, value: torch.Tensor = None):
        """Write to external memory if enabled."""
        if self.model.memory is not None:
            self.model.memory.write(key, value)
    
    def read_from_memory(self, query: torch.Tensor, top_k: int = 5):
        """Read from external memory if enabled."""
        if self.model.memory is not None:
            return self.model.memory.read(query, top_k)
        return None, None, None
