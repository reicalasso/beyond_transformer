"""
PULSE Speculative Decoding

Provides fast generation through speculation:
- Draft model generates candidates
- Target model verifies in parallel
- 2-3x speedup for generation
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeculativeDecoder:
    """
    Speculative Decoding for fast generation.
    
    Uses a small draft model to generate candidate tokens,
    then verifies with the target model in parallel.
    
    Args:
        target_model: Large target model
        draft_model: Small draft model
        num_speculative_tokens: Tokens to speculate per step
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        num_speculative_tokens: int = 4,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.num_speculative_tokens = num_speculative_tokens
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Generate with speculative decoding.
        
        Args:
            input_ids: Initial tokens [batch, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated tokens [batch, new_seq_len]
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        generated = input_ids.clone()
        
        while generated.shape[1] < max_length:
            # Step 1: Draft model generates speculative tokens
            draft_tokens, draft_probs = self._draft_generate(
                generated, self.num_speculative_tokens, temperature, top_k, top_p
            )
            
            # Step 2: Target model verifies all tokens in parallel
            candidate = torch.cat([generated, draft_tokens], dim=1)
            target_logits = self.target_model(candidate)["logits"]
            
            # Step 3: Verify and accept/reject
            accepted, num_accepted = self._verify_tokens(
                draft_tokens, draft_probs, target_logits, generated.shape[1],
                temperature, top_k, top_p
            )
            
            # Step 4: Update generated sequence
            generated = torch.cat([generated, accepted], dim=1)
            
            # If we rejected all, sample one token from target
            if num_accepted == 0:
                next_logits = target_logits[:, generated.shape[1] - 1, :]
                next_token = self._sample(next_logits, temperature, top_k, top_p)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated[:, :max_length]
    
    def _draft_generate(
        self,
        context: torch.Tensor,
        num_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate speculative tokens with draft model."""
        draft_tokens = []
        draft_probs = []
        
        current = context
        
        for _ in range(num_tokens):
            logits = self.draft_model(current)["logits"][:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Sample
            token = self._sample(logits, temperature, top_k, top_p)
            prob = probs.gather(-1, token)
            
            draft_tokens.append(token)
            draft_probs.append(prob)
            
            current = torch.cat([current, token], dim=1)
        
        return torch.cat(draft_tokens, dim=1), torch.cat(draft_probs, dim=1)
    
    def _verify_tokens(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_logits: torch.Tensor,
        context_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens against target model."""
        batch_size, num_draft = draft_tokens.shape
        
        accepted = []
        num_accepted = 0
        
        for i in range(num_draft):
            # Get target probability for draft token
            target_logits_i = target_logits[:, context_len + i, :]
            target_probs = F.softmax(target_logits_i / temperature, dim=-1)
            
            draft_token = draft_tokens[:, i:i+1]
            draft_prob = draft_probs[:, i:i+1]
            target_prob = target_probs.gather(-1, draft_token)
            
            # Accept with probability min(1, target_prob / draft_prob)
            accept_prob = (target_prob / draft_prob.clamp(min=1e-10)).clamp(max=1.0)
            accept = torch.rand_like(accept_prob) < accept_prob
            
            if accept.all():
                accepted.append(draft_token)
                num_accepted += 1
            else:
                # Rejection - sample from adjusted distribution
                adjusted_probs = F.relu(target_probs - draft_probs.unsqueeze(-1).expand_as(target_probs))
                adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
                
                # Sample from adjusted
                if adjusted_probs.sum() > 0:
                    resampled = torch.multinomial(adjusted_probs.squeeze(0), 1)
                    accepted.append(resampled.unsqueeze(0))
                    num_accepted += 1
                break
        
        if accepted:
            return torch.cat(accepted, dim=1), num_accepted
        else:
            return torch.empty(batch_size, 0, dtype=draft_tokens.dtype, device=draft_tokens.device), 0
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample from logits with temperature, top-k, top-p."""
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


class DraftModel(nn.Module):
    """
    Lightweight draft model for speculative decoding.
    
    A smaller version of the target model with:
    - Fewer layers
    - Smaller hidden size
    - Shared embeddings with target (optional)
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension (smaller than target)
        num_layers: Number of layers (fewer than target)
        num_heads: Number of attention heads
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        shared_embeddings: nn.Embedding = None,
    ):
        super().__init__()
        
        if shared_embeddings is not None:
            self.embeddings = shared_embeddings
            # Project if dimensions differ
            if shared_embeddings.embedding_dim != hidden_size:
                self.embed_proj = nn.Linear(shared_embeddings.embedding_dim, hidden_size)
            else:
                self.embed_proj = nn.Identity()
        else:
            self.embeddings = nn.Embedding(vocab_size, hidden_size)
            self.embed_proj = nn.Identity()
        
        # Simple transformer layers
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers)
        
        # LM head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Causal mask
        self.register_buffer("mask", None, persistent=False)
    
    def forward(self, input_ids: torch.Tensor) -> dict:
        """Forward pass."""
        seq_len = input_ids.shape[1]
        
        # Embeddings
        x = self.embeddings(input_ids)
        x = self.embed_proj(x)
        
        # Causal mask
        if self.mask is None or self.mask.shape[0] < seq_len:
            self.mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device),
                diagonal=1
            ).bool()
        
        # Transformer
        x = self.layers(x, src_mask=self.mask[:seq_len, :seq_len])
        
        # LM head
        logits = self.lm_head(x)
        
        return {"logits": logits}
