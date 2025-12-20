"""
PULSE Memory Service

Core memory management service that integrates with PULSE's hierarchical memory.
Provides working, short-term, and long-term memory layers with consolidation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import time
import uuid

import torch
import torch.nn as nn

from ..core.memory import HierarchicalMemory, MemoryBank


class MemoryLayer(str, Enum):
    """Memory layer types."""
    WORKING = "working"
    SHORT_TERM = "short-term"
    LONG_TERM = "long-term"
    AUTO = "auto"


class MemoryScope(str, Enum):
    """Memory scope/visibility."""
    PRIVATE = "private"
    SHARED = "shared"
    PUBLIC = "public"


@dataclass
class MemoryEntry:
    """Single memory entry."""
    id: str
    content: str
    embedding: torch.Tensor
    type: str
    scope: MemoryScope
    layer: MemoryLayer
    importance: float
    timestamp: float
    access_count: int
    last_access: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding embedding)."""
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "scope": self.scope.value,
            "layer": self.layer.value,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "metadata": self.metadata,
        }


class MemoryService(nn.Module):
    """
    PULSE Memory Service - MaaS Integration
    
    Manages hierarchical memory with:
    - Working memory: Active context (high-speed, full precision)
    - Short-term memory: Recent interactions (compressed)
    - Long-term memory: Persistent knowledge (highly compressed)
    
    Features:
    - Automatic consolidation (working → short-term → long-term)
    - Decay and forgetting of unimportant memories
    - Dynamic routing to relevant memories only
    - Semantic search and retrieval
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        working_slots: int = 32,
        short_term_slots: int = 128,
        long_term_slots: int = 512,
        consolidation_threshold: float = 0.7,
        decay_interval: int = 100,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.consolidation_threshold = consolidation_threshold
        self.decay_interval = decay_interval
        
        self.hierarchical_memory = HierarchicalMemory(
            hidden_size=hidden_size,
            working_slots=working_slots,
            short_term_slots=short_term_slots,
            long_term_slots=long_term_slots,
        )
        
        self.memory_store: Dict[str, MemoryEntry] = {}
        self.layer_indices: Dict[MemoryLayer, List[str]] = {
            MemoryLayer.WORKING: [],
            MemoryLayer.SHORT_TERM: [],
            MemoryLayer.LONG_TERM: [],
        }
        
        self.step_counter = 0
        
        self.embedding_proj = nn.Linear(hidden_size, hidden_size)
    
    def write_memory(
        self,
        content: str,
        embedding: torch.Tensor,
        memory_type: str = "general",
        scope: MemoryScope = MemoryScope.PRIVATE,
        layer: MemoryLayer = MemoryLayer.AUTO,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Write a new memory entry.
        
        Args:
            content: Text content of the memory
            embedding: Embedding vector [hidden_size]
            memory_type: Type of memory (e.g., 'preference', 'fact', 'conversation')
            scope: Memory visibility scope
            layer: Target memory layer (AUTO for automatic placement)
            importance: Optional importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        current_time = time.time()
        
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding_processed = self.embedding_proj(embedding)
        
        if importance is None:
            importance = self._calculate_importance(embedding_processed)
        
        if layer == MemoryLayer.AUTO:
            layer = self._determine_layer(importance)
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            embedding=embedding_processed,
            type=memory_type,
            scope=scope,
            layer=layer,
            importance=importance,
            timestamp=current_time,
            access_count=0,
            last_access=current_time,
            metadata=metadata or {},
        )
        
        self.memory_store[memory_id] = entry
        self.layer_indices[layer].append(memory_id)
        
        self._write_to_pulse_memory(entry)
        
        self.step_counter += 1
        if self.step_counter % self.decay_interval == 0:
            self._apply_decay()
        
        return memory_id
    
    def read_memory(
        self,
        query: str,
        query_embedding: torch.Tensor,
        limit: int = 5,
        layer: Optional[MemoryLayer] = None,
        memory_type: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
        min_importance: float = 0.0,
    ) -> List[MemoryEntry]:
        """
        Read memories matching the query.
        
        Args:
            query: Query text
            query_embedding: Query embedding vector
            limit: Maximum number of results
            layer: Filter by memory layer
            memory_type: Filter by memory type
            scope: Filter by scope
            min_importance: Minimum importance threshold
            
        Returns:
            List of matching memory entries
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        query_processed = self.embedding_proj(query_embedding)
        
        candidates = []
        for memory_id, entry in self.memory_store.items():
            if layer and entry.layer != layer:
                continue
            if memory_type and entry.type != memory_type:
                continue
            if scope and entry.scope != scope:
                continue
            if entry.importance < min_importance:
                continue
            
            similarity = self._compute_similarity(query_processed, entry.embedding)
            
            recency_boost = self._compute_recency_boost(entry.timestamp)
            access_boost = min(entry.access_count * 0.05, 0.3)
            
            score = similarity + recency_boost + access_boost
            
            candidates.append((score, entry))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, entry in candidates[:limit]:
            entry.access_count += 1
            entry.last_access = time.time()
            results.append(entry)
        
        return results
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory entry.
        
        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            importance: New importance score (optional)
            metadata: Metadata to merge (optional)
            
        Returns:
            Success status
        """
        if memory_id not in self.memory_store:
            return False
        
        entry = self.memory_store[memory_id]
        
        if content is not None:
            entry.content = content
        
        if importance is not None:
            entry.importance = importance
            new_layer = self._determine_layer(importance)
            if new_layer != entry.layer:
                self.layer_indices[entry.layer].remove(memory_id)
                entry.layer = new_layer
                self.layer_indices[new_layer].append(memory_id)
        
        if metadata is not None:
            entry.metadata.update(metadata)
        
        entry.last_access = time.time()
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            Success status
        """
        if memory_id not in self.memory_store:
            return False
        
        entry = self.memory_store[memory_id]
        self.layer_indices[entry.layer].remove(memory_id)
        del self.memory_store[memory_id]
        
        return True
    
    def consolidate_memories(self) -> Dict[str, int]:
        """
        Manually trigger memory consolidation.
        
        Moves memories between layers based on importance and access patterns.
        
        Returns:
            Statistics about consolidation
        """
        stats = {
            "working_to_short": 0,
            "short_to_long": 0,
            "forgotten": 0,
        }
        
        current_time = time.time()
        
        for memory_id in list(self.layer_indices[MemoryLayer.WORKING]):
            entry = self.memory_store[memory_id]
            age = current_time - entry.timestamp
            
            if age > 3600 or entry.importance > self.consolidation_threshold:
                self.layer_indices[MemoryLayer.WORKING].remove(memory_id)
                entry.layer = MemoryLayer.SHORT_TERM
                self.layer_indices[MemoryLayer.SHORT_TERM].append(memory_id)
                stats["working_to_short"] += 1
        
        for memory_id in list(self.layer_indices[MemoryLayer.SHORT_TERM]):
            entry = self.memory_store[memory_id]
            age = current_time - entry.timestamp
            
            if age > 86400 and entry.importance > 0.8:
                self.layer_indices[MemoryLayer.SHORT_TERM].remove(memory_id)
                entry.layer = MemoryLayer.LONG_TERM
                self.layer_indices[MemoryLayer.LONG_TERM].append(memory_id)
                stats["short_to_long"] += 1
        
        self.hierarchical_memory.working_memory.decay_memories()
        self.hierarchical_memory.short_term_memory.decay_memories()
        self.hierarchical_memory.long_term_memory.decay_memories()
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage."""
        return {
            "total_memories": len(self.memory_store),
            "working": len(self.layer_indices[MemoryLayer.WORKING]),
            "short_term": len(self.layer_indices[MemoryLayer.SHORT_TERM]),
            "long_term": len(self.layer_indices[MemoryLayer.LONG_TERM]),
            "total_accesses": sum(e.access_count for e in self.memory_store.values()),
            "avg_importance": sum(e.importance for e in self.memory_store.values()) / max(len(self.memory_store), 1),
        }
    
    def _calculate_importance(self, embedding: torch.Tensor) -> float:
        """Calculate importance score from embedding."""
        norm = embedding.norm(dim=-1).mean().item()
        return min(max(norm / 10.0, 0.0), 1.0)
    
    def _determine_layer(self, importance: float) -> MemoryLayer:
        """Determine appropriate layer based on importance."""
        if importance >= 0.8:
            return MemoryLayer.LONG_TERM
        elif importance >= 0.5:
            return MemoryLayer.SHORT_TERM
        else:
            return MemoryLayer.WORKING
    
    def _compute_similarity(self, query: torch.Tensor, memory: torch.Tensor) -> float:
        """Compute cosine similarity between query and memory."""
        query_norm = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        memory_norm = memory / (memory.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = (query_norm * memory_norm).sum(dim=-1).mean().item()
        return similarity
    
    def _compute_recency_boost(self, timestamp: float) -> float:
        """Compute recency boost for memory retrieval."""
        age = time.time() - timestamp
        return max(0.0, 0.3 * (1.0 - age / 86400.0))
    
    def _write_to_pulse_memory(self, entry: MemoryEntry):
        """Write entry to PULSE hierarchical memory."""
        if entry.embedding.dim() == 2:
            content = entry.embedding
        else:
            content = entry.embedding.unsqueeze(0)
        
        self.hierarchical_memory._consolidate(content)
    
    def _apply_decay(self):
        """Apply decay to old, unimportant memories."""
        current_time = time.time()
        to_delete = []
        
        for memory_id, entry in self.memory_store.items():
            age = current_time - entry.timestamp
            time_since_access = current_time - entry.last_access
            
            effective_importance = entry.importance * (0.99 ** (age / 3600))
            
            if effective_importance < 0.1 and time_since_access > 86400:
                to_delete.append(memory_id)
        
        for memory_id in to_delete:
            self.delete_memory(memory_id)
