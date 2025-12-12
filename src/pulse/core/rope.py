"""
PULSE Rotary Position Embeddings (RoPE)

Provides position encoding via rotation in complex space.
Benefits over sinusoidal:
- Better extrapolation to longer sequences
- Relative position awareness built-in
- No learned parameters needed
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position by rotating query/key vectors in 2D subspaces.
    Used in Llama, Mistral, GPT-NeoX, and other modern architectures.
    
    Args:
        dim: Head dimension (must be even)
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation (10000 default)
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build cos/sin cache
        self._build_cache(max_position_embeddings)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin for given sequence length.
        
        Args:
            seq_len: Current sequence length
            
        Returns:
            cos, sin tensors of shape [seq_len, dim]
        """
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len)
            self.max_position_embeddings = seq_len
        
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_heads, seq_len, head_dim]
        cos: Cosine cache [seq_len, head_dim]
        sin: Sine cache [seq_len, head_dim]
        position_ids: Optional position indices
        
    Returns:
        Rotated query and key tensors
    """
    # Reshape cos/sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RotaryEmbeddingScaled(RotaryEmbedding):
    """
    Scaled Rotary Embedding for better length extrapolation.
    
    Uses dynamic NTK-aware scaling when sequence exceeds training length.
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base)
    
    def _build_cache(self, seq_len: int):
        """Build cache with optional scaling."""
        if seq_len > self.max_position_embeddings:
            # Dynamic NTK-aware scaling
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) 
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq.to(self.inv_freq.device), persistent=False)
        
        super()._build_cache(seq_len)
