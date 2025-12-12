"""
PULSE Model - Main Language Model

A modular architecture using all PULSE components:
- RMSNorm for normalization
- RoPE for position encoding
- GQA for efficient attention
- SwiGLU for FFN
- Optional: SSM, MoE, Memory, Spiking
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.norm import RMSNorm
from ..core.attention import GroupedQueryAttention, StateAttention
from ..core.ffn import SwiGLU
from ..core.state import StateManager
from ..core.ssm import SSMBlock
from ..core.mixture import MixtureOfExperts, MixtureOfDepths


@dataclass
class PulseConfig:
    """PULSE model configuration."""
    
    # Model dimensions
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = None  # For GQA, defaults to num_heads // 4
    intermediate_size: int = None  # Defaults to hidden_size * 2.7
    
    # State configuration
    num_states: int = 32
    state_dim: int = None  # Defaults to hidden_size
    use_state_attention: bool = True
    
    # Optional modules
    use_ssm: bool = False  # Use SSM blocks
    use_moe: bool = False  # Use Mixture of Experts
    num_experts: int = 8
    top_k_experts: int = 2
    use_mod: bool = False  # Use Mixture of Depths
    mod_capacity: float = 0.5
    
    # Position and sequence
    max_position_embeddings: int = 8192
    rope_theta: float = 10000.0
    
    # Regularization
    dropout: float = 0.0
    
    # Other
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = max(self.num_heads // 4, 1)
        if self.intermediate_size is None:
            self.intermediate_size = int(self.hidden_size * 2.7)
        if self.state_dim is None:
            self.state_dim = self.hidden_size
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PulseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PulseEmbeddings(nn.Module):
    """Token embeddings (no position - using RoPE)."""
    
    def __init__(self, config: PulseConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class PulseLayer(nn.Module):
    """Single PULSE layer."""
    
    def __init__(self, config: PulseConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Self-attention with GQA and RoPE
        self.self_attn = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.dropout,
            max_position_embeddings=config.max_position_embeddings,
            use_rope=True,
        )
        
        # State attention (optional)
        if config.use_state_attention:
            self.state_attn = StateAttention(
                hidden_size=config.hidden_size,
                state_dim=config.state_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
        else:
            self.state_attn = None
        
        # FFN - either MoE or SwiGLU
        if config.use_moe and layer_idx % 2 == 1:  # MoE on odd layers
            self.ffn = MixtureOfExperts(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                top_k=config.top_k_experts,
            )
            self.is_moe = True
        else:
            self.ffn = SwiGLU(config.hidden_size, config.intermediate_size)
            self.is_moe = False
        
        # SSM block (optional, on specific layers)
        if config.use_ssm and layer_idx % 3 == 0:
            self.ssm = SSMBlock(config.hidden_size)
        else:
            self.ssm = None
        
        # Mixture of Depths (optional)
        if config.use_mod:
            self.mod = MixtureOfDepths(config.hidden_size, config.mod_capacity)
        else:
            self.mod = None
        
        # Norms
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.state_attn:
            self.state_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if self.ssm:
            self.ssm_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], float]:
        """
        Forward pass.
        
        Returns:
            hidden_states, updated_states, new_kv_cache, aux_loss
        """
        aux_loss = 0.0
        
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states, past_key_value=past_key_value, use_cache=use_cache
        )
        hidden_states = residual + hidden_states
        
        # State attention
        updated_states = states
        if self.state_attn is not None and states is not None:
            residual = hidden_states
            hidden_states = self.state_norm(hidden_states)
            state_out = self.state_attn(hidden_states, states)
            hidden_states = residual + state_out
        
        # SSM
        if self.ssm is not None:
            residual = hidden_states
            hidden_states = self.ssm_norm(hidden_states)
            hidden_states, _ = self.ssm(hidden_states)
            hidden_states = residual + hidden_states
        
        # FFN (with optional MoD)
        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        
        if self.mod is not None:
            hidden_states, _ = self.mod(hidden_states, self._ffn_forward)
        else:
            hidden_states = self._ffn_forward(hidden_states)
            if self.is_moe:
                hidden_states, moe_loss = hidden_states
                aux_loss += moe_loss
        
        hidden_states = residual + hidden_states
        
        return hidden_states, updated_states, new_kv, aux_loss
    
    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward for MoD compatibility."""
        return self.ffn(x)


class PulseModel(nn.Module):
    """PULSE base model."""
    
    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        
        # Embeddings
        self.embeddings = PulseEmbeddings(config)
        
        # State manager
        if config.use_state_attention:
            self.state_manager = StateManager(
                hidden_size=config.hidden_size,
                state_dim=config.state_dim,
                num_states=config.num_states,
            )
        else:
            self.state_manager = None
        
        # Layers
        self.layers = nn.ModuleList([
            PulseLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Initialize
        self.apply(self._init_weights)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory at the cost of speed."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
    
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
    ) -> Tuple[torch.Tensor, Optional[List], float]:
        """
        Forward pass.
        
        Returns:
            hidden_states, new_past_key_values, aux_loss
        """
        batch_size = input_ids.shape[0]
        
        # Embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Initialize states
        if self.state_manager is not None:
            states = self.state_manager.get_initial_states(batch_size)
        else:
            states = None
        
        # Process layers
        new_past_key_values = [] if use_cache else None
        total_aux_loss = 0.0
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training and not use_cache:
                # Use gradient checkpointing to save memory
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, states, new_kv, aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, states, past_kv, use_cache,
                    use_reentrant=False
                )
            else:
                hidden_states, states, new_kv, aux_loss = layer(
                    hidden_states, states, past_kv, use_cache
                )
            total_aux_loss += aux_loss
            
            if use_cache:
                new_past_key_values.append(new_kv)
            
            # Update states periodically
            if self.state_manager is not None and i % 3 == 2:
                states = self.state_manager(hidden_states, states)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, new_past_key_values, total_aux_loss


class PulseForCausalLM(nn.Module):
    """PULSE for causal language modeling."""
    
    def __init__(self, config: PulseConfig):
        super().__init__()
        self.config = config
        self.model = PulseModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.embed_tokens.weight
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.enable_gradient_checkpointing()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.model.disable_gradient_checkpointing()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        hidden_states, new_past_key_values, aux_loss = self.model(
            input_ids, past_key_values, use_cache
        )
        
        logits = self.lm_head(hidden_states)
        
        output = {"logits": logits}
        
        if use_cache:
            output["past_key_values"] = new_past_key_values
        
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss + aux_loss
            output["loss"] = loss
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = True,
    ) -> torch.Tensor:
        """
        Generate text with various decoding strategies.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_length: Maximum generation length
            min_length: Minimum generation length (EOS suppressed before this)
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            do_sample: Whether to sample (False = greedy/beam search)
            num_beams: Number of beams for beam search (1 = no beam search)
            early_stopping: Stop when all beams have finished
            
        Returns:
            Generated token IDs
        """
        if num_beams > 1:
            return self._beam_search(
                input_ids, max_length, num_beams, 
                eos_token_id, pad_token_id, early_stopping
            )
        
        return self._sample(
            input_ids, max_length, min_length, temperature,
            top_k, top_p, repetition_penalty, eos_token_id,
            pad_token_id, do_sample
        )
    
    @torch.no_grad()
    def _sample(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        min_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        do_sample: bool,
    ) -> torch.Tensor:
        """Sampling-based generation."""
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None
        
        # Track which sequences have finished
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        
        for step in range(max_length - input_ids.shape[1]):
            if not unfinished.any():
                break
                
            if past_key_values is None:
                outputs = self(generated, use_cache=True)
            else:
                outputs = self(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :].clone()
            
            # Suppress EOS before min_length
            cur_len = generated.shape[1]
            if eos_token_id is not None and cur_len < min_length:
                next_logits[:, eos_token_id] = float('-inf')
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token in set(generated[i].tolist()):
                        if next_logits[i, token] > 0:
                            next_logits[i, token] /= repetition_penalty
                        else:
                            next_logits[i, token] *= repetition_penalty
            
            if do_sample:
                # Temperature
                if temperature != 1.0 and temperature > 0:
                    next_logits = next_logits / temperature
                
                # Top-k
                if top_k > 0:
                    top_k_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0]
                    indices_to_remove = next_logits < top_k_vals[:, -1:]
                    next_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus sampling)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(next_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Handle finished sequences
            if pad_token_id is not None:
                next_token = next_token * unfinished.unsqueeze(-1) + pad_token_id * (~unfinished).unsqueeze(-1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update finished status
            if eos_token_id is not None:
                unfinished = unfinished & (next_token.squeeze(-1) != eos_token_id)
        
        return generated
    
    @torch.no_grad()
    def _beam_search(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        num_beams: int,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
        early_stopping: bool,
    ) -> torch.Tensor:
        """Beam search decoding."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        vocab_size = self.config.vocab_size
        
        # Expand input for beams
        input_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)
        
        # Initialize beam scores
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = float('-inf')  # Only first beam active initially
        beam_scores = beam_scores.view(-1)
        
        past_key_values = None
        done = [False] * batch_size
        generated_hyps = [[] for _ in range(batch_size)]
        
        for step in range(max_length - input_ids.shape[1] // num_beams):
            if all(done):
                break
            
            if past_key_values is None:
                outputs = self(input_ids, use_cache=True)
            else:
                outputs = self(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs["past_key_values"]
            next_logits = outputs["logits"][:, -1, :]
            
            # Compute scores
            next_scores = F.log_softmax(next_logits, dim=-1)
            next_scores = next_scores + beam_scores.unsqueeze(-1)
            
            # Reshape for beam selection
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams candidates
            next_scores, next_tokens = torch.topk(
                next_scores, 2 * num_beams, dim=-1, largest=True, sorted=True
            )
            
            # Compute beam indices and token indices
            next_beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Build next beams
            next_beam_scores = torch.zeros(batch_size, num_beams, device=device)
            next_beam_tokens = torch.zeros(batch_size, num_beams, dtype=torch.long, device=device)
            next_beam_indices_selected = torch.zeros(batch_size, num_beams, dtype=torch.long, device=device)
            
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_beam_scores[batch_idx] = beam_scores[batch_idx * num_beams:(batch_idx + 1) * num_beams]
                    next_beam_tokens[batch_idx] = pad_token_id if pad_token_id is not None else 0
                    next_beam_indices_selected[batch_idx] = torch.arange(num_beams, device=device)
                    continue
                
                beam_idx = 0
                for candidate_idx in range(2 * num_beams):
                    if beam_idx >= num_beams:
                        break
                    
                    score = next_scores[batch_idx, candidate_idx]
                    token = next_tokens[batch_idx, candidate_idx]
                    beam = next_beam_indices[batch_idx, candidate_idx]
                    
                    # Check if this is EOS
                    if eos_token_id is not None and token == eos_token_id:
                        generated_hyps[batch_idx].append((score.item(), input_ids[batch_idx * num_beams + beam].clone()))
                        if len(generated_hyps[batch_idx]) >= num_beams:
                            done[batch_idx] = True
                    else:
                        next_beam_scores[batch_idx, beam_idx] = score
                        next_beam_tokens[batch_idx, beam_idx] = token
                        next_beam_indices_selected[batch_idx, beam_idx] = beam
                        beam_idx += 1
            
            # Update beam scores
            beam_scores = next_beam_scores.view(-1)
            
            # Reorder past key values
            beam_idx_expanded = next_beam_indices_selected.view(-1)
            beam_idx_expanded = beam_idx_expanded + torch.arange(batch_size, device=device).unsqueeze(-1).expand(-1, num_beams).reshape(-1) * num_beams
            
            if past_key_values is not None:
                past_key_values = [
                    (k.index_select(0, beam_idx_expanded), v.index_select(0, beam_idx_expanded))
                    for k, v in past_key_values
                ]
            
            # Update input_ids
            input_ids = input_ids.index_select(0, beam_idx_expanded)
            input_ids = torch.cat([input_ids, next_beam_tokens.view(-1, 1)], dim=-1)
        
        # Select best hypothesis for each batch
        results = []
        for batch_idx in range(batch_size):
            if generated_hyps[batch_idx]:
                best_hyp = max(generated_hyps[batch_idx], key=lambda x: x[0])
                results.append(best_hyp[1])
            else:
                # Use best beam
                best_beam = beam_scores[batch_idx * num_beams:(batch_idx + 1) * num_beams].argmax()
                results.append(input_ids[batch_idx * num_beams + best_beam])
        
        # Pad to same length
        max_len = max(r.shape[0] for r in results)
        padded = torch.full((batch_size, max_len), pad_token_id if pad_token_id is not None else 0, 
                           dtype=torch.long, device=device)
        for i, r in enumerate(results):
            padded[i, :r.shape[0]] = r
        
        return padded
