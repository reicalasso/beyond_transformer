"""
PULSE KV Cache Management

Provides efficient KV caching for inference:
- KVCache: Standard KV cache
- CompressedKVCache: Compressed cache for long sequences
- SlidingWindowCache: Fixed-size sliding window
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """
    Standard Key-Value cache for autoregressive generation.
    
    Stores K and V tensors for each layer to avoid recomputation.
    """
    
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_len: int = 8192,
        num_heads: int = 8,
        head_dim: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache
        self.k_cache = torch.zeros(
            num_layers, max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_layers, max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a layer and return full K, V.
        
        Args:
            layer_idx: Layer index
            k: New keys [batch, num_heads, seq_len, head_dim]
            v: New values [batch, num_heads, seq_len, head_dim]
            
        Returns:
            Full K, V including cached values
        """
        batch_size, num_heads, new_seq_len, head_dim = k.shape
        
        # Store new values
        self.k_cache[layer_idx, :batch_size, :, self.seq_len:self.seq_len + new_seq_len] = k
        self.v_cache[layer_idx, :batch_size, :, self.seq_len:self.seq_len + new_seq_len] = v
        
        # Return full cache up to current position
        k_out = self.k_cache[layer_idx, :batch_size, :, :self.seq_len + new_seq_len]
        v_out = self.v_cache[layer_idx, :batch_size, :, :self.seq_len + new_seq_len]
        
        return k_out, v_out
    
    def advance(self, seq_len: int = 1):
        """Advance sequence position."""
        self.seq_len += seq_len
    
    def reset(self):
        """Reset cache."""
        self.seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


class CompressedKVCache:
    """
    Compressed KV Cache for very long sequences.
    
    Compresses old KV pairs to save memory while maintaining quality.
    Uses pooling to compress groups of tokens.
    
    Args:
        compression_ratio: How much to compress (4 = 4x compression)
        compression_start: Start compressing after this many tokens
    """
    
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int = 1,
        max_seq_len: int = 32768,
        num_heads: int = 8,
        head_dim: int = 64,
        compression_ratio: int = 4,
        compression_start: int = 2048,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.compression_ratio = compression_ratio
        self.compression_start = compression_start
        self.device = device
        self.dtype = dtype
        
        # Recent cache (uncompressed)
        self.recent_k = [None] * num_layers
        self.recent_v = [None] * num_layers
        
        # Compressed cache
        self.compressed_k = [None] * num_layers
        self.compressed_v = [None] * num_layers
        
        self.seq_len = 0
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full K, V."""
        # Add to recent cache
        if self.recent_k[layer_idx] is None:
            self.recent_k[layer_idx] = k
            self.recent_v[layer_idx] = v
        else:
            self.recent_k[layer_idx] = torch.cat([self.recent_k[layer_idx], k], dim=2)
            self.recent_v[layer_idx] = torch.cat([self.recent_v[layer_idx], v], dim=2)
        
        # Check if we need to compress
        recent_len = self.recent_k[layer_idx].shape[2]
        if recent_len > self.compression_start:
            self._compress(layer_idx)
        
        # Return combined cache
        if self.compressed_k[layer_idx] is not None:
            k_out = torch.cat([self.compressed_k[layer_idx], self.recent_k[layer_idx]], dim=2)
            v_out = torch.cat([self.compressed_v[layer_idx], self.recent_v[layer_idx]], dim=2)
        else:
            k_out = self.recent_k[layer_idx]
            v_out = self.recent_v[layer_idx]
        
        return k_out, v_out
    
    def _compress(self, layer_idx: int):
        """Compress old tokens."""
        recent_k = self.recent_k[layer_idx]
        recent_v = self.recent_v[layer_idx]
        
        # Keep last compression_start tokens uncompressed
        to_compress_k = recent_k[:, :, :-self.compression_start]
        to_compress_v = recent_v[:, :, :-self.compression_start]
        
        # Pool to compress
        batch, heads, seq_len, dim = to_compress_k.shape
        
        # Pad to multiple of compression_ratio
        pad_len = (self.compression_ratio - seq_len % self.compression_ratio) % self.compression_ratio
        if pad_len > 0:
            to_compress_k = F.pad(to_compress_k, (0, 0, 0, pad_len))
            to_compress_v = F.pad(to_compress_v, (0, 0, 0, pad_len))
        
        # Reshape and pool
        new_seq_len = (seq_len + pad_len) // self.compression_ratio
        compressed_k = to_compress_k.view(batch, heads, new_seq_len, self.compression_ratio, dim).mean(dim=3)
        compressed_v = to_compress_v.view(batch, heads, new_seq_len, self.compression_ratio, dim).mean(dim=3)
        
        # Update compressed cache
        if self.compressed_k[layer_idx] is not None:
            self.compressed_k[layer_idx] = torch.cat([self.compressed_k[layer_idx], compressed_k], dim=2)
            self.compressed_v[layer_idx] = torch.cat([self.compressed_v[layer_idx], compressed_v], dim=2)
        else:
            self.compressed_k[layer_idx] = compressed_k
            self.compressed_v[layer_idx] = compressed_v
        
        # Keep only recent uncompressed
        self.recent_k[layer_idx] = recent_k[:, :, -self.compression_start:]
        self.recent_v[layer_idx] = recent_v[:, :, -self.compression_start:]
    
    def reset(self):
        """Reset cache."""
        self.recent_k = [None] * self.num_layers
        self.recent_v = [None] * self.num_layers
        self.compressed_k = [None] * self.num_layers
        self.compressed_v = [None] * self.num_layers
        self.seq_len = 0


class SlidingWindowCache:
    """
    Sliding Window KV Cache.
    
    Only keeps the last N tokens, discarding older ones.
    Simple and memory-efficient for streaming.
    
    Args:
        window_size: Number of tokens to keep
    """
    
    def __init__(
        self,
        num_layers: int,
        window_size: int = 4096,
        max_batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.window_size = window_size
        self.device = device
        self.dtype = dtype
        
        # Circular buffer
        self.k_cache = torch.zeros(
            num_layers, max_batch_size, num_heads, window_size, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_layers, max_batch_size, num_heads, window_size, head_dim,
            device=device, dtype=dtype
        )
        
        self.position = 0
        self.filled = 0
    
    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with circular buffer."""
        batch_size, num_heads, seq_len, head_dim = k.shape
        
        for i in range(seq_len):
            pos = (self.position + i) % self.window_size
            self.k_cache[layer_idx, :batch_size, :, pos] = k[:, :, i]
            self.v_cache[layer_idx, :batch_size, :, pos] = v[:, :, i]
        
        # Return valid portion
        valid_len = min(self.filled + seq_len, self.window_size)
        
        if valid_len < self.window_size:
            k_out = self.k_cache[layer_idx, :batch_size, :, :valid_len]
            v_out = self.v_cache[layer_idx, :batch_size, :, :valid_len]
        else:
            # Need to reorder for correct sequence
            start = (self.position + seq_len) % self.window_size
            indices = torch.arange(self.window_size, device=self.device)
            indices = (indices + start) % self.window_size
            k_out = self.k_cache[layer_idx, :batch_size, :, indices]
            v_out = self.v_cache[layer_idx, :batch_size, :, indices]
        
        return k_out, v_out
    
    def advance(self, seq_len: int = 1):
        """Advance position."""
        self.position = (self.position + seq_len) % self.window_size
        self.filled = min(self.filled + seq_len, self.window_size)
    
    def reset(self):
        """Reset cache."""
        self.position = 0
        self.filled = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
