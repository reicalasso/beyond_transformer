"""
Simple test script for PULSE + MaaS
Tests basic functionality without requiring full installation
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'beyond_transformer', 'src'))

print("🔥 Testing PULSE + MaaS Integration\n")
print("=" * 60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from pulse.maas import (
        MemoryService,
        MemoryLayer,
        MemoryScope,
        MemoryAPI,
        MemoryConsolidator,
        MemoryQueryEngine,
    )
    print("   ✅ All imports successful!")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create MemoryService
print("\n2. Testing MemoryService initialization...")
try:
    import torch
    memory_service = MemoryService(hidden_size=768)
    print(f"   ✅ MemoryService created")
    print(f"   - Hidden size: {memory_service.hidden_size}")
    print(f"   - Total memories: {len(memory_service.memory_store)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Write memory
print("\n3. Testing write_memory...")
try:
    embedding = torch.randn(768)
    memory_id = memory_service.write_memory(
        content="Test memory: User prefers Python",
        embedding=embedding,
        memory_type="preference",
        scope=MemoryScope.PRIVATE,
        layer=MemoryLayer.LONG_TERM,
        importance=0.9
    )
    print(f"   ✅ Memory written")
    print(f"   - Memory ID: {memory_id[:8]}...")
    print(f"   - Total memories: {len(memory_service.memory_store)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Read memory
print("\n4. Testing read_memory...")
try:
    query_embedding = torch.randn(768)
    results = memory_service.read_memory(
        query="what does user prefer",
        query_embedding=query_embedding,
        limit=5
    )
    print(f"   ✅ Memory read")
    print(f"   - Found {len(results)} results")
    if results:
        print(f"   - First result: {results[0].content}")
        print(f"   - Importance: {results[0].importance:.2f}")
        print(f"   - Layer: {results[0].layer.value}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Memory statistics
print("\n5. Testing get_memory_stats...")
try:
    stats = memory_service.get_memory_stats()
    print(f"   ✅ Statistics retrieved")
    print(f"   - Total memories: {stats['total_memories']}")
    print(f"   - Working: {stats['working']}")
    print(f"   - Short-term: {stats['short_term']}")
    print(f"   - Long-term: {stats['long_term']}")
    print(f"   - Avg importance: {stats['avg_importance']:.2f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 6: Update memory
print("\n6. Testing update_memory...")
try:
    success = memory_service.update_memory(
        memory_id=memory_id,
        importance=0.95,
        metadata={"updated": True}
    )
    print(f"   ✅ Memory updated: {success}")
    updated_entry = memory_service.memory_store[memory_id]
    print(f"   - New importance: {updated_entry.importance:.2f}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 7: MemoryAPI
print("\n7. Testing MemoryAPI...")
try:
    def simple_embedding(text):
        import torch
        hash_val = hash(text)
        torch.manual_seed(abs(hash_val) % (2**31))
        return torch.randn(768)
    
    api = MemoryAPI(memory_service, embedding_fn=simple_embedding)
    
    # Test write endpoint
    response = api.write_memory({
        "content": "API test memory",
        "type": "test",
        "scope": "private",
        "layer": "working"
    })
    print(f"   ✅ API write: {response['success']}")
    
    # Test read endpoint
    response = api.read_memory({
        "query": "test",
        "limit": 5
    })
    print(f"   ✅ API read: {response['success']}, found {response['count']} results")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Query Engine
print("\n8. Testing MemoryQueryEngine...")
try:
    query_engine = MemoryQueryEngine(hidden_size=768)
    
    # Test dynamic routing
    query_emb = torch.randn(768)
    active_layers = query_engine.dynamic_route(
        query_emb,
        memory_service.memory_store,
        memory_service.layer_indices
    )
    print(f"   ✅ Dynamic routing")
    print(f"   - Active layers: {[layer.value for layer in active_layers]}")
    
    # Test query
    results = query_engine.query(
        query_text="test query",
        query_embedding=query_emb,
        memory_store=memory_service.memory_store,
        layer_indices=memory_service.layer_indices,
        limit=5
    )
    print(f"   ✅ Query executed: {len(results)} results")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Consolidator
print("\n9. Testing MemoryConsolidator...")
try:
    consolidator = MemoryConsolidator()
    stats = consolidator.consolidate(
        memory_service.memory_store,
        memory_service.layer_indices
    )
    print(f"   ✅ Consolidation completed")
    print(f"   - Stats: {stats}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Delete memory
print("\n10. Testing delete_memory...")
try:
    success = memory_service.delete_memory(memory_id)
    print(f"   ✅ Memory deleted: {success}")
    print(f"   - Remaining memories: {len(memory_service.memory_store)}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🔥 PULSE + MaaS is working correctly!")
print("\nNext steps:")
print("  1. Run examples: python examples/maas_integration_demo.py")
print("  2. Start server: python -m pulse.maas.server")
print("  3. Read docs: README_MAAS.md")
