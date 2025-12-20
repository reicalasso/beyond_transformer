"""
Memory Query Engine

Advanced query processing for memory retrieval:
- Semantic search with embeddings
- Multi-layer querying
- Dynamic routing to relevant memories
- Query expansion and refinement
"""

from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_service import MemoryEntry, MemoryLayer, MemoryScope


class MemoryQueryEngine:
    """
    Advanced query engine for memory retrieval.
    
    Features:
    - Semantic similarity search
    - Multi-layer parallel querying
    - Dynamic routing (only activate relevant layers)
    - Query expansion
    - Result re-ranking
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        top_k_per_layer: int = 5,
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
    ):
        """
        Initialize query engine.
        
        Args:
            hidden_size: Embedding dimension
            top_k_per_layer: Number of results per layer
            enable_query_expansion: Enable query expansion
            enable_reranking: Enable result re-ranking
        """
        self.hidden_size = hidden_size
        self.top_k_per_layer = top_k_per_layer
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
    
    def query(
        self,
        query_text: str,
        query_embedding: torch.Tensor,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
        limit: int = 10,
        layers: Optional[List[MemoryLayer]] = None,
        memory_type: Optional[str] = None,
        scope: Optional[MemoryScope] = None,
        min_importance: float = 0.0,
    ) -> List[Tuple[float, MemoryEntry]]:
        """
        Execute advanced memory query.
        
        Args:
            query_text: Query text
            query_embedding: Query embedding vector
            memory_store: All memories
            layer_indices: Memory indices by layer
            limit: Maximum results
            layers: Target layers (None = all)
            memory_type: Filter by type
            scope: Filter by scope
            min_importance: Minimum importance
            
        Returns:
            List of (score, entry) tuples
        """
        if layers is None:
            layers = [MemoryLayer.WORKING, MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM]
        
        if self.enable_query_expansion:
            expanded_queries = self._expand_query(query_embedding)
        else:
            expanded_queries = [query_embedding]
        
        all_candidates = []
        
        for layer in layers:
            layer_candidates = self._query_layer(
                expanded_queries,
                memory_store,
                layer_indices[layer],
                layer,
                memory_type,
                scope,
                min_importance,
            )
            all_candidates.extend(layer_candidates)
        
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        
        if self.enable_reranking and len(all_candidates) > limit:
            all_candidates = self._rerank_results(
                query_embedding,
                all_candidates[:limit * 2]
            )
        
        return all_candidates[:limit]
    
    def dynamic_route(
        self,
        query_embedding: torch.Tensor,
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
    ) -> List[MemoryLayer]:
        """
        Dynamically determine which layers to query.
        
        Only activates layers that are likely to contain relevant memories.
        This is the "Dynamic Routing" feature - like activating brain regions.
        
        Args:
            query_embedding: Query embedding
            memory_store: All memories
            layer_indices: Memory indices by layer
            
        Returns:
            List of layers to query
        """
        layer_scores = {}
        
        for layer in [MemoryLayer.WORKING, MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM]:
            if not layer_indices[layer]:
                layer_scores[layer] = 0.0
                continue
            
            sample_size = min(10, len(layer_indices[layer]))
            sample_ids = layer_indices[layer][:sample_size]
            
            similarities = []
            for memory_id in sample_ids:
                entry = memory_store[memory_id]
                sim = self._compute_similarity(query_embedding, entry.embedding)
                similarities.append(sim)
            
            layer_scores[layer] = max(similarities) if similarities else 0.0
        
        active_layers = []
        threshold = 0.3
        
        if layer_scores[MemoryLayer.WORKING] > threshold:
            active_layers.append(MemoryLayer.WORKING)
        if layer_scores[MemoryLayer.SHORT_TERM] > threshold * 0.8:
            active_layers.append(MemoryLayer.SHORT_TERM)
        if layer_scores[MemoryLayer.LONG_TERM] > threshold * 0.6:
            active_layers.append(MemoryLayer.LONG_TERM)
        
        if not active_layers:
            active_layers = [MemoryLayer.WORKING]
        
        return active_layers
    
    def _query_layer(
        self,
        query_embeddings: List[torch.Tensor],
        memory_store: Dict[str, MemoryEntry],
        memory_ids: List[str],
        layer: MemoryLayer,
        memory_type: Optional[str],
        scope: Optional[MemoryScope],
        min_importance: float,
    ) -> List[Tuple[float, MemoryEntry]]:
        """Query a single memory layer."""
        candidates = []
        
        for memory_id in memory_ids:
            entry = memory_store[memory_id]
            
            if memory_type and entry.type != memory_type:
                continue
            if scope and entry.scope != scope:
                continue
            if entry.importance < min_importance:
                continue
            
            max_similarity = 0.0
            for query_emb in query_embeddings:
                sim = self._compute_similarity(query_emb, entry.embedding)
                max_similarity = max(max_similarity, sim)
            
            recency_boost = self._compute_recency_boost(entry, layer)
            access_boost = min(entry.access_count * 0.03, 0.2)
            importance_boost = entry.importance * 0.2
            
            score = max_similarity + recency_boost + access_boost + importance_boost
            
            candidates.append((score, entry))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:self.top_k_per_layer]
    
    def _expand_query(
        self,
        query_embedding: torch.Tensor,
        num_expansions: int = 2,
    ) -> List[torch.Tensor]:
        """
        Expand query with variations.
        
        Creates slight variations of the query embedding to improve recall.
        """
        queries = [query_embedding]
        
        for _ in range(num_expansions):
            noise = torch.randn_like(query_embedding) * 0.1
            expanded = query_embedding + noise
            expanded = expanded / (expanded.norm(dim=-1, keepdim=True) + 1e-8)
            queries.append(expanded)
        
        return queries
    
    def _rerank_results(
        self,
        query_embedding: torch.Tensor,
        candidates: List[Tuple[float, MemoryEntry]],
    ) -> List[Tuple[float, MemoryEntry]]:
        """
        Re-rank results using more sophisticated scoring.
        
        Considers:
        - Semantic similarity
        - Diversity (avoid redundant results)
        - Importance
        - Recency
        """
        if not candidates:
            return candidates
        
        reranked = []
        selected_embeddings = []
        
        for score, entry in candidates:
            base_similarity = self._compute_similarity(
                query_embedding, entry.embedding
            )
            
            diversity_penalty = 0.0
            for selected_emb in selected_embeddings:
                sim_to_selected = self._compute_similarity(
                    entry.embedding, selected_emb
                )
                diversity_penalty += max(0, sim_to_selected - 0.7) * 0.3
            
            final_score = (
                base_similarity * 0.5 +
                entry.importance * 0.3 +
                min(entry.access_count * 0.02, 0.2) -
                diversity_penalty
            )
            
            reranked.append((final_score, entry))
            selected_embeddings.append(entry.embedding)
        
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked
    
    def _compute_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity."""
        if emb1.dim() == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.dim() == 1:
            emb2 = emb2.unsqueeze(0)
        
        emb1_norm = emb1 / (emb1.norm(dim=-1, keepdim=True) + 1e-8)
        emb2_norm = emb2 / (emb2.norm(dim=-1, keepdim=True) + 1e-8)
        
        similarity = (emb1_norm * emb2_norm).sum(dim=-1).mean().item()
        return similarity
    
    def _compute_recency_boost(
        self,
        entry: MemoryEntry,
        layer: MemoryLayer,
    ) -> float:
        """Compute recency boost based on layer."""
        import time
        age = time.time() - entry.timestamp
        
        if layer == MemoryLayer.WORKING:
            return max(0.0, 0.4 * (1.0 - age / 3600.0))
        elif layer == MemoryLayer.SHORT_TERM:
            return max(0.0, 0.3 * (1.0 - age / 86400.0))
        else:
            return max(0.0, 0.1 * (1.0 - age / 604800.0))
    
    def batch_query(
        self,
        queries: List[Tuple[str, torch.Tensor]],
        memory_store: Dict[str, MemoryEntry],
        layer_indices: Dict[MemoryLayer, List[str]],
        limit: int = 10,
    ) -> List[List[Tuple[float, MemoryEntry]]]:
        """
        Execute multiple queries in batch.
        
        Args:
            queries: List of (text, embedding) tuples
            memory_store: All memories
            layer_indices: Memory indices by layer
            limit: Results per query
            
        Returns:
            List of result lists
        """
        results = []
        
        for query_text, query_embedding in queries:
            query_results = self.query(
                query_text=query_text,
                query_embedding=query_embedding,
                memory_store=memory_store,
                layer_indices=layer_indices,
                limit=limit,
            )
            results.append(query_results)
        
        return results
