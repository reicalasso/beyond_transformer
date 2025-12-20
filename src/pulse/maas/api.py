"""
PULSE Memory API

RESTful API endpoints for Memory-as-a-Service.
Provides HTTP interface for memory operations.
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json

from .memory_service import MemoryService, MemoryLayer, MemoryScope, MemoryEntry
import torch


class MemoryAPI:
    """
    Memory-as-a-Service API
    
    Provides HTTP-compatible interface for memory operations:
    - POST /pulse/memory/write - Write new memory
    - POST /pulse/memory/read - Query memories
    - PUT /pulse/memory/update - Update existing memory
    - DELETE /pulse/memory/delete - Delete memory
    - POST /pulse/memory/consolidate - Trigger consolidation
    - GET /pulse/memory/stats - Get memory statistics
    """
    
    def __init__(
        self,
        memory_service: MemoryService,
        embedding_fn: Optional[callable] = None,
    ):
        """
        Initialize API.
        
        Args:
            memory_service: MemoryService instance
            embedding_fn: Function to convert text to embeddings
        """
        self.memory_service = memory_service
        self.embedding_fn = embedding_fn or self._default_embedding
    
    def write_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write memory endpoint.
        
        POST /pulse/memory/write
        {
            "content": "User prefers Rust",
            "type": "preference",
            "scope": "private",
            "layer": "long-term",
            "importance": 0.9,
            "metadata": {"category": "programming"}
        }
        
        Returns:
        {
            "success": true,
            "memory_id": "uuid",
            "layer": "long-term"
        }
        """
        try:
            content = request.get("content")
            if not content:
                return {"success": False, "error": "Content is required"}
            
            memory_type = request.get("type", "general")
            scope = MemoryScope(request.get("scope", "private"))
            layer = MemoryLayer(request.get("layer", "auto"))
            importance = request.get("importance")
            metadata = request.get("metadata", {})
            
            embedding = self.embedding_fn(content)
            
            memory_id = self.memory_service.write_memory(
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                scope=scope,
                layer=layer,
                importance=importance,
                metadata=metadata,
            )
            
            entry = self.memory_service.memory_store[memory_id]
            
            return {
                "success": True,
                "memory_id": memory_id,
                "layer": entry.layer.value,
                "importance": entry.importance,
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read memory endpoint.
        
        POST /pulse/memory/read
        {
            "query": "what language does user prefer",
            "limit": 3,
            "layer": "preference",
            "type": "preference",
            "min_importance": 0.5
        }
        
        Returns:
        {
            "success": true,
            "results": [
                {
                    "id": "uuid",
                    "content": "User prefers Rust",
                    "type": "preference",
                    "importance": 0.9,
                    "score": 0.95,
                    ...
                }
            ],
            "count": 1
        }
        """
        try:
            query = request.get("query")
            if not query:
                return {"success": False, "error": "Query is required"}
            
            limit = request.get("limit", 5)
            layer_str = request.get("layer")
            layer = MemoryLayer(layer_str) if layer_str else None
            memory_type = request.get("type")
            scope_str = request.get("scope")
            scope = MemoryScope(scope_str) if scope_str else None
            min_importance = request.get("min_importance", 0.0)
            
            query_embedding = self.embedding_fn(query)
            
            results = self.memory_service.read_memory(
                query=query,
                query_embedding=query_embedding,
                limit=limit,
                layer=layer,
                memory_type=memory_type,
                scope=scope,
                min_importance=min_importance,
            )
            
            return {
                "success": True,
                "results": [entry.to_dict() for entry in results],
                "count": len(results),
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def update_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update memory endpoint.
        
        PUT /pulse/memory/update
        {
            "memory_id": "uuid",
            "content": "Updated content",
            "importance": 0.95,
            "metadata": {"updated": true}
        }
        
        Returns:
        {
            "success": true,
            "memory_id": "uuid"
        }
        """
        try:
            memory_id = request.get("memory_id")
            if not memory_id:
                return {"success": False, "error": "memory_id is required"}
            
            content = request.get("content")
            importance = request.get("importance")
            metadata = request.get("metadata")
            
            success = self.memory_service.update_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                metadata=metadata,
            )
            
            if success:
                return {"success": True, "memory_id": memory_id}
            else:
                return {"success": False, "error": "Memory not found"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def delete_memory(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete memory endpoint.
        
        DELETE /pulse/memory/delete
        {
            "memory_id": "uuid"
        }
        
        Returns:
        {
            "success": true,
            "memory_id": "uuid"
        }
        """
        try:
            memory_id = request.get("memory_id")
            if not memory_id:
                return {"success": False, "error": "memory_id is required"}
            
            success = self.memory_service.delete_memory(memory_id)
            
            if success:
                return {"success": True, "memory_id": memory_id}
            else:
                return {"success": False, "error": "Memory not found"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def consolidate_memories(self, request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Consolidate memories endpoint.
        
        POST /pulse/memory/consolidate
        {}
        
        Returns:
        {
            "success": true,
            "stats": {
                "working_to_short": 5,
                "short_to_long": 2,
                "forgotten": 3
            }
        }
        """
        try:
            stats = self.memory_service.consolidate_memories()
            
            return {
                "success": True,
                "stats": stats,
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_stats(self, request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get memory statistics endpoint.
        
        GET /pulse/memory/stats
        
        Returns:
        {
            "success": true,
            "stats": {
                "total_memories": 100,
                "working": 20,
                "short_term": 50,
                "long_term": 30,
                ...
            }
        }
        """
        try:
            stats = self.memory_service.get_memory_stats()
            
            return {
                "success": True,
                "stats": stats,
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _default_embedding(self, text: str) -> torch.Tensor:
        """
        Default embedding function (simple hash-based).
        In production, use a proper embedding model.
        """
        hash_val = hash(text)
        torch.manual_seed(abs(hash_val) % (2**31))
        embedding = torch.randn(self.memory_service.hidden_size)
        return embedding
    
    def handle_request(self, endpoint: str, method: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route requests to appropriate handlers.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            request: Request body
            
        Returns:
            Response dictionary
        """
        routes = {
            ("/pulse/memory/write", "POST"): self.write_memory,
            ("/pulse/memory/read", "POST"): self.read_memory,
            ("/pulse/memory/update", "PUT"): self.update_memory,
            ("/pulse/memory/delete", "DELETE"): self.delete_memory,
            ("/pulse/memory/consolidate", "POST"): self.consolidate_memories,
            ("/pulse/memory/stats", "GET"): self.get_stats,
        }
        
        handler = routes.get((endpoint, method))
        if handler:
            return handler(request)
        else:
            return {
                "success": False,
                "error": f"Unknown endpoint: {method} {endpoint}"
            }
