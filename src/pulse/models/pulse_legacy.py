"""
PULSE model v1 - legacy language model.

DEPRECATED: Use the current `PulseConfig` / `PulseModel` / `PulseForCausalLM`
interfaces instead. This module is kept for backward compatibility only.

Simple GQA + SwiGLU architecture.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.norm import RMSNorm
from ..core.attention import GroupedQueryAttention
from ..core.ffn import SwiGLU


@dataclass
class PulseConfig:
    """PULSE v1 model configuration (legacy)."""
    
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = None
    intermediate_size: int = None
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = max(self.num_heads // 4, 1)
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 2.7)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PulseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PulseLayer(nn.Module):
    """Single PULSE v1 layer - simple GQA + SwiGLU."""
    
    def __init__(self, config: PulseConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            use_rope=True,
        )
        self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Attention
        residual = x
        x = self.input_norm(x)
        x, new_kv = self.self_attn(x, past_key_value=past_key_value, use_cache=use_cache)
        x = residual + x
        
        # FFN
        residual = x
        x = self.post_attn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x, new_kv


class PulseModel(nn.Module):
    """PULSE v1 base model (legacy)."""
    
    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([PulseLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        x = self.embed(input_ids)
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x, new_kv = layer(x, past_kv, use_cache)
            if use_cache:
                new_past_key_values.append(new_kv)
        
        x = self.norm(x)
        return x, new_past_key_values


class PulseForCausalLM(nn.Module):
    """PULSE v1 for causal language modeling (legacy)."""
    
    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.model = PulseModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        hidden, new_kv = self.model(input_ids, past_key_values, use_cache)
        logits = self.lm_head(hidden)
        
        output = {"logits": logits}
        if use_cache:
            output["past_key_values"] = new_kv
        
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
        """Simple autoregressive generation."""
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            outputs = self(generated if past_key_values is None else generated[:, -1:],
                          past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)
            
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                next_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
