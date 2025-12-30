"""
PULSE Simple Memory

LRU-style fixed-size memory cache.
Replaces complex 3-tier hierarchical memory with single efficient buffer.

Features:
- Fixed capacity with circular buffer
- Similarity-based retrieval
- O(k) lookup where k = capacity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMemory(nn.Module):
    """
    Fixed-size LRU memory cache.
    
    Simple and efficient memory for storing/retrieving embeddings.
    Uses cosine similarity for retrieval and circular buffer for updates.
    
    Args:
        hidden_size: Embedding dimension
        capacity: Maximum number of memory slots
    """
    
    def __init__(self, hidden_size: int, capacity: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity = capacity
        
        # Memory buffers (non-learnable)
        self.register_buffer('keys', torch.zeros(capacity, hidden_size))
        self.register_buffer('values', torch.zeros(capacity, hidden_size))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('size', torch.zeros(1, dtype=torch.long))
        
        # Projection for query
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def write(self, key: torch.Tensor, value: torch.Tensor = None) -> int:
        """
        Write to memory (circular buffer style).
        
        Args:
            key: Key embedding [hidden_size] or [batch, hidden_size]
            value: Value embedding (defaults to key)
            
        Returns:
            Index where written
        """
        if value is None:
            value = key
        
        # Handle batch dimension
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        
        batch_size = key.shape[0]
        
        for i in range(batch_size):
            idx = self.ptr.item()
            self.keys[idx] = key[i].detach()
            self.values[idx] = value[i].detach()
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, torch.tensor([self.capacity], device=self.size.device))
        
        return idx
    
    def read(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Read from memory using similarity search.
        
        Args:
            query: Query embedding [hidden_size] or [batch, hidden_size]
            top_k: Number of results to return
            
        Returns:
            values: Retrieved values [batch, top_k, hidden_size]
            scores: Similarity scores [batch, top_k]
            indices: Memory indices [batch, top_k]
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        batch_size = query.shape[0]
        current_size = self.size.item()
        
        if current_size == 0:
            # Empty memory
            return (
                torch.zeros(batch_size, top_k, self.hidden_size, device=query.device),
                torch.zeros(batch_size, top_k, device=query.device),
                torch.zeros(batch_size, top_k, dtype=torch.long, device=query.device),
            )
        
        # Project query
        query_proj = self.query_proj(query)
        
        # Compute similarities
        keys_norm = F.normalize(self.keys[:current_size], dim=-1)
        query_norm = F.normalize(query_proj, dim=-1)
        
        similarities = query_norm @ keys_norm.T  # [batch, size]
        
        # Get top-k
        k = min(top_k, current_size)
        scores, indices = similarities.topk(k, dim=-1)
        
        # Gather values
        values = self.values[indices]  # [batch, k, hidden]
        
        # Pad if needed
        if k < top_k:
            pad_size = top_k - k
            values = F.pad(values, (0, 0, 0, pad_size))
            scores = F.pad(scores, (0, pad_size), value=-1)
            indices = F.pad(indices, (0, pad_size), value=-1)
        
        return values, scores, indices
    
    def read_attend(
        self,
        query: torch.Tensor,
        top_k: int = 16,
    ) -> torch.Tensor:
        """
        Read from memory with soft attention.
        
        Args:
            query: Query embedding [batch, seq_len, hidden_size]
            top_k: Number of memories to attend to
            
        Returns:
            Attended memory output [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = query.shape
        
        # Pool query for retrieval
        query_pooled = query.mean(dim=1)  # [batch, hidden]
        
        # Get top-k memories
        values, scores, _ = self.read(query_pooled, top_k)
        
        if scores.max() < 0:
            # No valid memories
            return torch.zeros_like(query)
        
        # Attend to retrieved memories
        query_proj = self.query_proj(query)  # [batch, seq, hidden]
        
        # Attention over retrieved memories
        attn_scores = torch.bmm(query_proj, values.transpose(1, 2))  # [batch, seq, k]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.bmm(attn_weights, values)  # [batch, seq, hidden]
        
        return output
    
    def clear(self):
        """Clear all memory."""
        self.keys.zero_()
        self.values.zero_()
        self.ptr.zero_()
        self.size.zero_()
    
    def __len__(self) -> int:
        return self.size.item()


class MemoryAugmentedBlock(nn.Module):
    """
    Block that can read/write to external memory.
    
    Optional wrapper for UnifiedBlock to add memory capabilities.
    
    Args:
        hidden_size: Model dimension
        memory_capacity: Memory size
    """
    
    def __init__(
        self,
        hidden_size: int,
        memory_capacity: int = 512,
    ):
        super().__init__()
        self.memory = SimpleMemory(hidden_size, memory_capacity)
        self.gate = nn.Linear(hidden_size * 2, hidden_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        write: bool = False,
    ) -> torch.Tensor:
        """
        Process with memory augmentation.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            write: Whether to write to memory
            
        Returns:
            Output [batch, seq_len, hidden_size]
        """
        # Read from memory
        mem_output = self.memory.read_attend(x)
        
        # Gated fusion
        combined = torch.cat([x, mem_output], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        output = gate * x + (1 - gate) * mem_output
        
        # Optionally write to memory
        if write:
            pooled = x.mean(dim=1)  # [batch, hidden]
            self.memory.write(pooled)
        
        return output
