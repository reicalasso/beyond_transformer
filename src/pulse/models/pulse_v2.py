"""
PULSE v2 - Minimal Language Model

Radically simplified architecture:
- UnifiedBlock: Single primitive for local+global processing
- RecurrentState: Single compressed state vector
- SimpleMemory: Optional LRU cache

No SSM/MoE/MoD complexity. Just what works.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    memory_capacity: int = 512
    use_recurrent_state: bool = True
    
    # Regularization
    dropout: float = 0.0
    
    # Other
    norm_eps: float = 1e-6
    tie_embeddings: bool = True
    pad_token_id: int = 0
    
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
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
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
        
        # Optional recurrent state
        if config.use_recurrent_state:
            self.recurrent = RecurrentState(config.hidden_size)
        else:
            self.recurrent = None
        
        # Optional external key–value memory
        if config.use_memory:
            self.memory = KeyValueMemory(config.hidden_size, config.memory_capacity)
        else:
            self.memory = None
        
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
        attention_state: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            recurrent_state: Optional [batch, hidden_size]
            
        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            new_recurrent_state: Updated recurrent state
            layer_states: List of per-layer linear attention states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed
        x = self.embed(input_ids)
        x = self.dropout(x)
        
        # Build padding mask if provided: [batch, seq_len] -> [batch, 1, 1, seq_len]
        if attention_mask is not None:
            # Mask: 1 for tokens, 0 for padding
            pad_mask = attention_mask.view(batch_size, 1, 1, seq_len)
        else:
            pad_mask = None
        
        # Initialize recurrent state
        if self.recurrent is not None and recurrent_state is None:
            recurrent_state = self.recurrent.get_initial_state(batch_size)
        
        # Process layers
        layer_states: List[torch.Tensor] = []
        if attention_state is None:
            attention_state = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, layer_state = layer(x, state=attention_state[i], attention_mask=pad_mask)
            layer_states.append(layer_state)
        
        # Update recurrent state (after all layers)
        if self.recurrent is not None:
            recurrent_state = self.recurrent(x, recurrent_state)
        
        # Memory augmentation (optional)
        if self.memory is not None:
            mem_output = self.memory.read_attend(x)
            x = x + 0.1 * mem_output  # Small residual from memory
        
        # Final norm
        x = self.norm(x)
        
        return x, recurrent_state, layer_states


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
        attention_state: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            labels: Optional [batch, seq_len] for training
            recurrent_state: Optional recurrent state
            
        Returns:
            Dict with logits, loss (if labels), and recurrent_state
        """
        hidden, recurrent_state, layer_states = self.model(
            input_ids,
            recurrent_state=recurrent_state,
            attention_state=attention_state,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden)
        
        output = {
            "logits": logits,
            "recurrent_state": recurrent_state,
            "attention_state": layer_states,
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
    ) -> torch.Tensor:
        """
        Simple autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            eos_token_id: Stop token
            
        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        generated = input_ids.clone()
        recurrent_state = None
        attention_state: Optional[List[torch.Tensor]] = None
        attention_mask: Optional[torch.Tensor] = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            if attention_mask is None:
                attention_mask = (generated != self.config.pad_token_id).long()
            
            outputs = self(
                generated,
                recurrent_state=recurrent_state,
                attention_state=attention_state,
                attention_mask=attention_mask,
            )
            recurrent_state = outputs["recurrent_state"]
            attention_state = outputs.get("attention_state")
            
            # Get next token logits
            next_logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, (next_token != self.config.pad_token_id).long()],
                dim=1,
            )
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def write_to_memory(self, key: torch.Tensor, value: torch.Tensor = None):
        """Write to external memory if enabled."""
        if self.model.memory is not None:
            self.model.memory.write(key, value)
    
    def read_from_memory(self, query: torch.Tensor, top_k: int = 5):
        """Read from external memory if enabled."""
        if self.model.memory is not None:
            return self.model.memory.read(query, top_k)
        return None, None, None
