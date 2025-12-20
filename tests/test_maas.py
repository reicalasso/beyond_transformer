"""
Tests for PULSE + MaaS Integration
"""

import pytest
import torch
from pulse.maas import (
    MemoryService,
    MemoryLayer,
    MemoryScope,
    MemoryAPI,
)
from pulse.maas.consolidation import MemoryConsolidator
from pulse.maas.query_engine import MemoryQueryEngine


def simple_embedding(text: str, dim: int = 768) -> torch.Tensor:
    """Simple embedding function for testing."""
    hash_val = hash(text)
    torch.manual_seed(abs(hash_val) % (2**31))
    return torch.randn(dim)


class TestMemoryService:
    """Test MemoryService functionality."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = MemoryService(hidden_size=768)
        assert service.hidden_size == 768
        assert len(service.memory_store) == 0
    
    def test_write_memory(self):
        """Test writing memories."""
        service = MemoryService(hidden_size=768)
        
        memory_id = service.write_memory(
            content="Test memory",
            embedding=simple_embedding("Test memory"),
            memory_type="test",
            scope=MemoryScope.PRIVATE,
            layer=MemoryLayer.WORKING,
        )
        
        assert memory_id in service.memory_store
        assert len(service.memory_store) == 1
        assert service.memory_store[memory_id].content == "Test memory"
    
    def test_read_memory(self):
        """Test reading memories."""
        service = MemoryService(hidden_size=768)
        
        service.write_memory(
            content="Python is great",
            embedding=simple_embedding("Python is great"),
            memory_type="preference",
            scope=MemoryScope.PRIVATE,
            layer=MemoryLayer.LONG_TERM,
            importance=0.9,
        )
        
        results = service.read_memory(
            query="programming language",
            query_embedding=simple_embedding("programming language"),
            limit=5,
        )
        
        assert len(results) > 0
        assert results[0].content == "Python is great"
    
    def test_update_memory(self):
        """Test updating memories."""
        service = MemoryService(hidden_size=768)
        
        memory_id = service.write_memory(
            content="Original content",
            embedding=simple_embedding("Original content"),
            memory_type="test",
            scope=MemoryScope.PRIVATE,
            layer=MemoryLayer.WORKING,
            importance=0.5,
        )
        
        success = service.update_memory(
            memory_id=memory_id,
            importance=0.9,
        )
        
        assert success
        assert service.memory_store[memory_id].importance == 0.9
    
    def test_delete_memory(self):
        """Test deleting memories."""
        service = MemoryService(hidden_size=768)
        
        memory_id = service.write_memory(
            content="To be deleted",
            embedding=simple_embedding("To be deleted"),
            memory_type="test",
            scope=MemoryScope.PRIVATE,
            layer=MemoryLayer.WORKING,
        )
        
        assert memory_id in service.memory_store
        
        success = service.delete_memory(memory_id)
        
        assert success
        assert memory_id not in service.memory_store
    
    def test_memory_layers(self):
        """Test memory layer assignment."""
        service = MemoryService(hidden_size=768)
        
        service.write_memory(
            content="Working memory",
            embedding=simple_embedding("Working memory"),
            memory_type="test",
            layer=MemoryLayer.WORKING,
        )
        
        service.write_memory(
            content="Short-term memory",
            embedding=simple_embedding("Short-term memory"),
            memory_type="test",
            layer=MemoryLayer.SHORT_TERM,
        )
        
        service.write_memory(
            content="Long-term memory",
            embedding=simple_embedding("Long-term memory"),
            memory_type="test",
            layer=MemoryLayer.LONG_TERM,
        )
        
        assert len(service.layer_indices[MemoryLayer.WORKING]) == 1
        assert len(service.layer_indices[MemoryLayer.SHORT_TERM]) == 1
        assert len(service.layer_indices[MemoryLayer.LONG_TERM]) == 1
    
    def test_get_stats(self):
        """Test getting memory statistics."""
        service = MemoryService(hidden_size=768)
        
        for i in range(5):
            service.write_memory(
                content=f"Memory {i}",
                embedding=simple_embedding(f"Memory {i}"),
                memory_type="test",
                layer=MemoryLayer.WORKING,
            )
        
        stats = service.get_memory_stats()
        
        assert stats["total_memories"] == 5
        assert stats["working"] == 5


class TestMemoryAPI:
    """Test MemoryAPI functionality."""
    
    def test_write_endpoint(self):
        """Test write memory endpoint."""
        service = MemoryService(hidden_size=768)
        api = MemoryAPI(service, embedding_fn=simple_embedding)
        
        request = {
            "content": "API test memory",
            "type": "test",
            "scope": "private",
            "layer": "working",
        }
        
        response = api.write_memory(request)
        
        assert response["success"] is True
        assert "memory_id" in response
    
    def test_read_endpoint(self):
        """Test read memory endpoint."""
        service = MemoryService(hidden_size=768)
        api = MemoryAPI(service, embedding_fn=simple_embedding)
        
        api.write_memory({
            "content": "Test content",
            "type": "test",
        })
        
        response = api.read_memory({
            "query": "test",
            "limit": 5,
        })
        
        assert response["success"] is True
        assert response["count"] > 0
    
    def test_update_endpoint(self):
        """Test update memory endpoint."""
        service = MemoryService(hidden_size=768)
        api = MemoryAPI(service, embedding_fn=simple_embedding)
        
        write_response = api.write_memory({
            "content": "Original",
            "type": "test",
        })
        
        memory_id = write_response["memory_id"]
        
        update_response = api.update_memory({
            "memory_id": memory_id,
            "importance": 0.95,
        })
        
        assert update_response["success"] is True
    
    def test_delete_endpoint(self):
        """Test delete memory endpoint."""
        service = MemoryService(hidden_size=768)
        api = MemoryAPI(service, embedding_fn=simple_embedding)
        
        write_response = api.write_memory({
            "content": "To delete",
            "type": "test",
        })
        
        memory_id = write_response["memory_id"]
        
        delete_response = api.delete_memory({
            "memory_id": memory_id,
        })
        
        assert delete_response["success"] is True


class TestMemoryConsolidator:
    """Test MemoryConsolidator functionality."""
    
    def test_consolidation(self):
        """Test memory consolidation."""
        service = MemoryService(hidden_size=768)
        consolidator = MemoryConsolidator(
            working_ttl=0.1,
            short_term_ttl=0.2,
        )
        
        for i in range(5):
            service.write_memory(
                content=f"Memory {i}",
                embedding=simple_embedding(f"Memory {i}"),
                memory_type="test",
                layer=MemoryLayer.WORKING,
                importance=0.8,
            )
        
        import time
        time.sleep(0.15)
        
        stats = consolidator.consolidate(
            service.memory_store,
            service.layer_indices,
        )
        
        assert stats["working_to_short"] > 0


class TestQueryEngine:
    """Test MemoryQueryEngine functionality."""
    
    def test_query(self):
        """Test basic query."""
        service = MemoryService(hidden_size=768)
        query_engine = MemoryQueryEngine(hidden_size=768)
        
        service.write_memory(
            content="Python programming",
            embedding=simple_embedding("Python programming"),
            memory_type="test",
            layer=MemoryLayer.WORKING,
        )
        
        results = query_engine.query(
            query_text="programming",
            query_embedding=simple_embedding("programming"),
            memory_store=service.memory_store,
            layer_indices=service.layer_indices,
            limit=5,
        )
        
        assert len(results) > 0
    
    def test_dynamic_routing(self):
        """Test dynamic routing."""
        service = MemoryService(hidden_size=768)
        query_engine = MemoryQueryEngine(hidden_size=768)
        
        service.write_memory(
            content="Working memory content",
            embedding=simple_embedding("Working memory content"),
            memory_type="test",
            layer=MemoryLayer.WORKING,
        )
        
        active_layers = query_engine.dynamic_route(
            simple_embedding("working memory"),
            service.memory_store,
            service.layer_indices,
        )
        
        assert len(active_layers) > 0
        assert MemoryLayer.WORKING in active_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
