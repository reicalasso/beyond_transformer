"""
PULSE Hierarchical Memory System

Implements biologically-inspired memory management:
- Short-term memory (GPU, high precision)
- Long-term memory (compressed, lower precision)
- Memory consolidation and decay
- Adaptive recall based on relevance
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    Hierarchical memory bank with compression and decay.
    
    Inspired by human memory:
    - Working memory: Recent, high-fidelity (GPU)
    - Long-term memory: Compressed, decaying importance
    """
    
    def __init__(
        self,
        memory_dim: int,
        num_slots: int = 64,
        compression_ratio: float = 0.5,
        decay_rate: float = 0.99,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        self.compression_ratio = compression_ratio
        self.decay_rate = decay_rate
        
        # Compressed dimension for long-term storage
        self.compressed_dim = int(memory_dim * compression_ratio)
        
        # Memory slots (learnable initial state)
        self.register_buffer('memory', torch.zeros(1, num_slots, memory_dim))
        self.register_buffer('importance', torch.ones(1, num_slots))
        self.register_buffer('age', torch.zeros(1, num_slots))
        
        # Compression/decompression networks
        self.compressor = nn.Linear(memory_dim, self.compressed_dim, bias=False)
        self.decompressor = nn.Linear(self.compressed_dim, memory_dim, bias=False)
        
        # Importance scoring
        self.importance_net = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 4),
            nn.ReLU(),
            nn.Linear(memory_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Query mechanism for recall
        self.query_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        self.key_proj = nn.Linear(memory_dim, memory_dim, bias=False)
    
    def write(
        self,
        content: torch.Tensor,
        force_slot: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Write content to memory with importance-based slot selection.
        
        Args:
            content: [batch, dim] content to write
            force_slot: Optional specific slot to write to
            
        Returns:
            Updated memory state
        """
        batch_size = content.shape[0]
        
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1).clone()
        importance = self.importance.expand(batch_size, -1).clone()
        age = self.age.expand(batch_size, -1).clone()
        
        # Calculate importance of new content
        new_importance = self.importance_net(content).squeeze(-1)  # [batch]
        
        if force_slot is not None:
            slot_idx = force_slot
        else:
            # Find least important slot (considering age decay)
            effective_importance = importance * (self.decay_rate ** age)
            slot_idx = effective_importance.argmin(dim=-1)  # [batch]
        
        # Write to selected slots
        for b in range(batch_size):
            idx = slot_idx if isinstance(slot_idx, int) else slot_idx[b].item()
            memory[b, idx] = content[b]
            importance[b, idx] = new_importance[b] if new_importance.dim() > 0 else new_importance
            age[b, idx] = 0
        
        # Age all memories
        age = age + 1
        
        # Return updated memory state (don't modify buffers during forward pass)
        # Buffers are only initial states; actual state should be tracked externally
        return memory, importance, age
    
    def read(
        self,
        query: torch.Tensor,
        top_k: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using attention-based recall.
        
        Args:
            query: [batch, seq_len, dim] query vectors
            top_k: Number of memory slots to attend to
            
        Returns:
            Retrieved content and attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)
        importance = self.importance.expand(batch_size, -1)
        age = self.age.expand(batch_size, -1)
        
        # Project query and keys
        q = self.query_proj(query)  # [batch, seq_len, dim]
        k = self.key_proj(memory)   # [batch, num_slots, dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.memory_dim)
        
        # Modulate by importance and recency
        effective_importance = importance * (self.decay_rate ** age)
        scores = scores + effective_importance.unsqueeze(1).log().clamp(min=-10)
        
        # Sparse attention: only attend to top-k slots
        if top_k < self.num_slots:
            topk_scores, topk_indices = scores.topk(top_k, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Retrieve content
        retrieved = torch.matmul(attn_weights, memory)
        
        return retrieved, attn_weights
    
    def compress_and_store(self, content: torch.Tensor) -> torch.Tensor:
        """Compress content for long-term storage."""
        return self.compressor(content)
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress content from long-term storage."""
        return self.decompressor(compressed)
    
    def decay_memories(self):
        """Apply memory decay - forget unimportant old memories."""
        effective_importance = self.importance * (self.decay_rate ** self.age)
        
        # Reset very old, unimportant memories
        reset_mask = effective_importance < 0.1
        self.memory = self.memory * (~reset_mask).unsqueeze(-1).float()
        self.importance = self.importance.masked_fill(reset_mask, 1.0)
        self.age = self.age.masked_fill(reset_mask, 0)


class HierarchicalMemory(nn.Module):
    """
    Multi-level hierarchical memory system.
    
    Levels:
    1. Working memory: Current context (full precision)
    2. Short-term memory: Recent past (compressed)
    3. Long-term memory: Distant past (highly compressed, sparse)
    """
    
    def __init__(
        self,
        hidden_size: int,
        working_slots: int = 32,
        short_term_slots: int = 128,
        long_term_slots: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Working memory (full precision, fast access)
        self.working_memory = MemoryBank(
            memory_dim=hidden_size,
            num_slots=working_slots,
            compression_ratio=1.0,
            decay_rate=0.95,
        )
        
        # Short-term memory (compressed)
        self.short_term_memory = MemoryBank(
            memory_dim=hidden_size,
            num_slots=short_term_slots,
            compression_ratio=0.5,
            decay_rate=0.99,
        )
        
        # Long-term memory (highly compressed)
        self.long_term_memory = MemoryBank(
            memory_dim=hidden_size,
            num_slots=long_term_slots,
            compression_ratio=0.25,
            decay_rate=0.999,
        )
        
        # Consolidation gate (decides what to move to long-term)
        self.consolidation_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size * 3, hidden_size)
    
    def forward(
        self,
        query: torch.Tensor,
        new_content: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Query all memory levels and optionally write new content.
        
        Args:
            query: [batch, seq_len, hidden_size]
            new_content: Optional new content to store
            
        Returns:
            Retrieved and combined memory content
        """
        # Read from all levels
        working_out, _ = self.working_memory.read(query, top_k=8)
        short_term_out, _ = self.short_term_memory.read(query, top_k=16)
        long_term_out, _ = self.long_term_memory.read(query, top_k=32)
        
        # Combine outputs
        combined = torch.cat([working_out, short_term_out, long_term_out], dim=-1)
        output = self.output_proj(combined)
        
        # Write new content if provided
        if new_content is not None:
            self._consolidate(new_content)
        
        return output
    
    def _consolidate(self, content: torch.Tensor):
        """
        Consolidate memories: move from working to short-term to long-term.
        """
        # Always write to working memory
        if content.dim() == 3:
            content = content.mean(dim=1)  # Pool sequence dimension
        
        # Write returns (memory, importance, age) tuple now
        _ = self.working_memory.write(content)
        
        # Decide what to consolidate to short-term
        consolidate_score = self.consolidation_gate(content)
        if consolidate_score.mean() > 0.5:
            compressed = self.short_term_memory.compress_and_store(content)
            _ = self.short_term_memory.write(
                self.short_term_memory.decompressor(compressed)
            )
        
        # Periodically consolidate to long-term
        self.working_memory.decay_memories()
        self.short_term_memory.decay_memories()
        self.long_term_memory.decay_memories()


class StreamingContext(nn.Module):
    """
    Streaming context manager for infinite context length.
    
    Maintains a compressed summary of past context that can be
    queried without reprocessing the entire history.
    """
    
    def __init__(
        self,
        hidden_size: int,
        summary_size: int = 256,
        chunk_size: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.summary_size = summary_size
        self.chunk_size = chunk_size
        
        # Summary state (compressed representation of all past)
        self.register_buffer('summary', torch.zeros(1, summary_size, hidden_size))
        
        # Summary update network (GRU-style)
        self.summary_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.summary_update = nn.Linear(hidden_size * 2, hidden_size)
        
        # Query the summary
        self.summary_query = nn.MultiheadAttention(
            hidden_size, num_heads=8, batch_first=True
        )
    
    def update_summary(self, new_content: torch.Tensor):
        """
        Update the running summary with new content.
        
        Args:
            new_content: [batch, seq_len, hidden_size]
        """
        batch_size = new_content.shape[0]
        
        # Pool new content
        pooled = new_content.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        # Expand summary for batch
        summary = self.summary.expand(batch_size, -1, -1)
        
        # GRU-style update for each summary slot
        combined = torch.cat([summary, pooled.expand(-1, self.summary_size, -1)], dim=-1)
        gate = torch.sigmoid(self.summary_gate(combined))
        update = torch.tanh(self.summary_update(combined))
        
        new_summary = gate * summary + (1 - gate) * update
        
        # Update buffer (use mean across batch for stability)
        self.summary = new_summary.mean(dim=0, keepdim=True)
    
    def query_summary(self, query: torch.Tensor) -> torch.Tensor:
        """
        Query the summary for relevant past context.
        
        Args:
            query: [batch, seq_len, hidden_size]
            
        Returns:
            Retrieved context from summary
        """
        batch_size = query.shape[0]
        summary = self.summary.expand(batch_size, -1, -1)
        
        output, _ = self.summary_query(query, summary, summary)
        return output
    
    def process_stream(
        self,
        input_stream: torch.Tensor,
        process_fn: callable,
    ) -> torch.Tensor:
        """
        Process a long input stream in chunks, maintaining context.
        
        Args:
            input_stream: [batch, long_seq_len, hidden_size]
            process_fn: Function to process each chunk
            
        Returns:
            Processed output with full context awareness
        """
        batch_size, seq_len, _ = input_stream.shape
        outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            chunk = input_stream[:, i:i+self.chunk_size, :]
            
            # Get context from summary
            context = self.query_summary(chunk)
            
            # Process chunk with context
            chunk_with_context = chunk + context
            output = process_fn(chunk_with_context)
            
            # Update summary with this chunk
            self.update_summary(chunk)
            
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)
