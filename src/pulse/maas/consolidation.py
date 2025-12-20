"""
Memory Consolidation Engine

Implements intelligent memory consolidation strategies:
- Time-based consolidation (working → short-term → long-term)
- Importance-based promotion
- Access pattern analysis
- Semantic clustering
"""

from typing import Dict, List, Set, Tuple
import time
from collections import defaultdict

import torch
import torch.nn as nn

from .memory_service import MemoryEntry, MemoryLayer


class MemoryConsolidator:
    """
    Intelligent memory consolidation engine.
    
    Consolidates memories based on:
    - Time decay
    - Access patterns
    - Importance scores
    - Semantic similarity
    """
    
    def __init__(
        self,
        working_ttl: float = 3600.0,
        short_term_ttl: float = 86400.0,
        importance_threshold: float = 0.7,
        access_threshold: int = 5,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize consolidator.
        
        Args:
            working_ttl: Time-to-live for working memory (seconds)
            short_term_ttl: Time-to-live for short-term memory (seconds)
            importance_threshold: Importance threshold for promotion
            access_threshold: Access count threshold for promotion
            similarity_threshold: Similarity threshold for clustering
        """
        self.working_ttl = working_ttl
        self.short_term_ttl = short_term_ttl
        self.importance_threshold = importance_threshold
        self.access_threshold = access_threshold
        self.similarity_threshold = similarity_threshold
    
    def consolidate(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
    ) -> Dict[str, int]:
        """
        Perform memory consolidation.
        
        Args:
            memory_store: Dictionary of all memories
            layer_indices: Indices of memories by layer
            
        Returns:
            Consolidation statistics
        """
        stats = {
            "working_to_short": 0,
            "short_to_long": 0,
            "promoted_by_importance": 0,
            "promoted_by_access": 0,
            "clustered": 0,
            "forgotten": 0,
        }
        
        current_time = time.time()
        
        stats.update(self._consolidate_working(
            memory_store, layer_indices, current_time
        ))
        
        stats.update(self._consolidate_short_term(
            memory_store, layer_indices, current_time
        ))
        
        stats.update(self._promote_by_importance(
            memory_store, layer_indices
        ))
        
        stats.update(self._promote_by_access(
            memory_store, layer_indices
        ))
        
        stats.update(self._cluster_similar_memories(
            memory_store, layer_indices
        ))
        
        return stats
    
    def _consolidate_working(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
        current_time: float,
    ) -> Dict[str, int]:
        """Consolidate working memory to short-term."""
        stats = {"working_to_short": 0}
        
        to_promote = []
        for memory_id in layer_indices[MemoryLayer.WORKING]:
            entry = memory_store[memory_id]
            age = current_time - entry.timestamp
            
            if age > self.working_ttl:
                to_promote.append(memory_id)
            elif entry.importance > self.importance_threshold:
                to_promote.append(memory_id)
        
        for memory_id in to_promote:
            entry = memory_store[memory_id]
            layer_indices[MemoryLayer.WORKING].remove(memory_id)
            entry.layer = MemoryLayer.SHORT_TERM
            layer_indices[MemoryLayer.SHORT_TERM].append(memory_id)
            stats["working_to_short"] += 1
        
        return stats
    
    def _consolidate_short_term(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
        current_time: float,
    ) -> Dict[str, int]:
        """Consolidate short-term memory to long-term."""
        stats = {"short_to_long": 0}
        
        to_promote = []
        for memory_id in layer_indices[MemoryLayer.SHORT_TERM]:
            entry = memory_store[memory_id]
            age = current_time - entry.timestamp
            
            if age > self.short_term_ttl and entry.importance > 0.8:
                to_promote.append(memory_id)
            elif entry.access_count > self.access_threshold * 2:
                to_promote.append(memory_id)
        
        for memory_id in to_promote:
            entry = memory_store[memory_id]
            layer_indices[MemoryLayer.SHORT_TERM].remove(memory_id)
            entry.layer = MemoryLayer.LONG_TERM
            layer_indices[MemoryLayer.LONG_TERM].append(memory_id)
            stats["short_to_long"] += 1
        
        return stats
    
    def _promote_by_importance(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
    ) -> Dict[str, int]:
        """Promote memories based on importance."""
        stats = {"promoted_by_importance": 0}
        
        for memory_id in list(layer_indices[MemoryLayer.WORKING]):
            entry = memory_store[memory_id]
            if entry.importance > 0.9:
                layer_indices[MemoryLayer.WORKING].remove(memory_id)
                entry.layer = MemoryLayer.LONG_TERM
                layer_indices[MemoryLayer.LONG_TERM].append(memory_id)
                stats["promoted_by_importance"] += 1
        
        for memory_id in list(layer_indices[MemoryLayer.SHORT_TERM]):
            entry = memory_store[memory_id]
            if entry.importance > 0.95:
                layer_indices[MemoryLayer.SHORT_TERM].remove(memory_id)
                entry.layer = MemoryLayer.LONG_TERM
                layer_indices[MemoryLayer.LONG_TERM].append(memory_id)
                stats["promoted_by_importance"] += 1
        
        return stats
    
    def _promote_by_access(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
    ) -> Dict[str, int]:
        """Promote frequently accessed memories."""
        stats = {"promoted_by_access": 0}
        
        for memory_id in list(layer_indices[MemoryLayer.WORKING]):
            entry = memory_store[memory_id]
            if entry.access_count > self.access_threshold:
                layer_indices[MemoryLayer.WORKING].remove(memory_id)
                entry.layer = MemoryLayer.SHORT_TERM
                layer_indices[MemoryLayer.SHORT_TERM].append(memory_id)
                entry.importance = min(entry.importance + 0.1, 1.0)
                stats["promoted_by_access"] += 1
        
        for memory_id in list(layer_indices[MemoryLayer.SHORT_TERM]):
            entry = memory_store[memory_id]
            if entry.access_count > self.access_threshold * 3:
                layer_indices[MemoryLayer.SHORT_TERM].remove(memory_id)
                entry.layer = MemoryLayer.LONG_TERM
                layer_indices[MemoryLayer.LONG_TERM].append(memory_id)
                entry.importance = min(entry.importance + 0.1, 1.0)
                stats["promoted_by_access"] += 1
        
        return stats
    
    def _cluster_similar_memories(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
    ) -> Dict[str, int]:
        """Cluster and merge similar memories."""
        stats = {"clustered": 0}
        
        for layer in [MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM]:
            memory_ids = layer_indices[layer]
            if len(memory_ids) < 2:
                continue
            
            clusters = self._find_clusters(memory_store, memory_ids)
            
            for cluster in clusters:
                if len(cluster) > 1:
                    representative = self._merge_cluster(
                        memory_store, cluster
                    )
                    stats["clustered"] += len(cluster) - 1
        
        return stats
    
    def _find_clusters(
        self,
        memory_store: Dict[str, MemoryEntry],
        memory_ids: List[str],
    ) -> List[List[str]]:
        """Find clusters of similar memories."""
        clusters = []
        visited = set()
        
        for i, id1 in enumerate(memory_ids):
            if id1 in visited:
                continue
            
            cluster = [id1]
            visited.add(id1)
            
            entry1 = memory_store[id1]
            
            for id2 in memory_ids[i+1:]:
                if id2 in visited:
                    continue
                
                entry2 = memory_store[id2]
                
                if entry1.type == entry2.type:
                    similarity = self._compute_similarity(
                        entry1.embedding, entry2.embedding
                    )
                    
                    if similarity > self.similarity_threshold:
                        cluster.append(id2)
                        visited.add(id2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _merge_cluster(
        self,
        memory_store: Dict[str, MemoryEntry],
        cluster: List[str],
    ) -> str:
        """Merge a cluster of similar memories."""
        entries = [memory_store[mid] for mid in cluster]
        
        most_important = max(entries, key=lambda e: e.importance)
        
        total_accesses = sum(e.access_count for e in entries)
        most_important.access_count = total_accesses
        
        avg_importance = sum(e.importance for e in entries) / len(entries)
        most_important.importance = max(most_important.importance, avg_importance)
        
        most_important.metadata["merged_from"] = [e.id for e in entries if e.id != most_important.id]
        
        return most_important.id
    
    def _compute_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between embeddings."""
        emb1_norm = emb1 / (emb1.norm(dim=-1, keepdim=True) + 1e-8)
        emb2_norm = emb2 / (emb2.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = (emb1_norm * emb2_norm).sum(dim=-1).mean().item()
        return similarity
    
    def apply_decay(
        self,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
        decay_rate: float = 0.99,
    ) -> Dict[str, int]:
        """
        Apply decay to memories.
        
        Args:
            memory_store: Dictionary of all memories
            layer_indices: Indices of memories by layer
            decay_rate: Decay rate per time unit
            
        Returns:
            Decay statistics
        """
        stats = {"forgotten": 0}
        
        current_time = time.time()
        to_delete = []
        
        for memory_id, entry in memory_store.items():
            age = current_time - entry.timestamp
            time_since_access = current_time - entry.last_access
            
            age_hours = age / 3600.0
            effective_importance = entry.importance * (decay_rate ** age_hours)
            
            if effective_importance < 0.05 and time_since_access > 86400:
                to_delete.append(memory_id)
            else:
                entry.importance = effective_importance
        
        for memory_id in to_delete:
            entry = memory_store[memory_id]
            layer_indices[entry.layer].remove(memory_id)
            del memory_store[memory_id]
            stats["forgotten"] += 1
        
        return stats
