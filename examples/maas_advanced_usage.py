"""
PULSE + MaaS Advanced Usage Example

Demonstrates advanced features:
- Dynamic routing
- Memory consolidation
- Query expansion
- Multi-layer querying
"""

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
    """Simple embedding function."""
    hash_val = hash(text)
    torch.manual_seed(abs(hash_val) % (2**31))
    return torch.randn(dim)


def main():
    print("🔥 PULSE + MaaS Advanced Usage Example\n")
    
    memory_service = MemoryService(hidden_size=768)
    consolidator = MemoryConsolidator()
    query_engine = MemoryQueryEngine(hidden_size=768)
    
    print("=" * 60)
    print("1. Populating Memory with Various Types")
    print("=" * 60)
    
    memories = [
        ("User loves Python and uses it daily", "preference", 0.9, "long-term"),
        ("User is building a chatbot with transformers", "fact", 0.8, "short-term"),
        ("User asked about attention mechanisms", "conversation", 0.6, "working"),
        ("User prefers functional programming", "preference", 0.85, "long-term"),
        ("User mentioned using PyTorch yesterday", "fact", 0.7, "short-term"),
        ("User is interested in reinforcement learning", "interest", 0.75, "short-term"),
        ("User completed a project on image classification", "achievement", 0.9, "long-term"),
        ("User asked about CUDA optimization", "conversation", 0.5, "working"),
    ]
    
    for content, mem_type, importance, layer in memories:
        embedding = simple_embedding(content)
        memory_id = memory_service.write_memory(
            content=content,
            embedding=embedding,
            memory_type=mem_type,
            scope=MemoryScope.PRIVATE,
            layer=MemoryLayer(layer),
            importance=importance,
        )
        print(f"✓ Written: {content[:50]}... (Layer: {layer})")
    
    print(f"\n✓ Total memories: {len(memory_service.memory_store)}\n")
    
    print("=" * 60)
    print("2. Dynamic Routing - Smart Layer Selection")
    print("=" * 60)
    
    queries = [
        "what does user prefer",
        "current conversation topics",
        "user's technical skills",
    ]
    
    for query in queries:
        query_emb = simple_embedding(query)
        active_layers = query_engine.dynamic_route(
            query_emb,
            memory_service.memory_store,
            memory_service.layer_indices,
        )
        print(f"✓ Query: '{query}'")
        print(f"  Active layers: {[layer.value for layer in active_layers]}")
    print()
    
    print("=" * 60)
    print("3. Advanced Query with Re-ranking")
    print("=" * 60)
    
    query = "what is user working on and interested in"
    query_emb = simple_embedding(query)
    
    results = query_engine.query(
        query_text=query,
        query_embedding=query_emb,
        memory_store=memory_service.memory_store,
        layer_indices=memory_service.layer_indices,
        limit=5,
    )
    
    print(f"✓ Query: '{query}'")
    print(f"✓ Found {len(results)} results:\n")
    for i, (score, entry) in enumerate(results, 1):
        print(f"  {i}. {entry.content}")
        print(f"     Score: {score:.3f}, Layer: {entry.layer.value}, "
              f"Importance: {entry.importance:.2f}")
    print()
    
    print("=" * 60)
    print("4. Memory Consolidation")
    print("=" * 60)
    
    stats = consolidator.consolidate(
        memory_service.memory_store,
        memory_service.layer_indices,
    )
    
    print(f"✓ Consolidation Statistics:")
    for key, value in stats.items():
        if value > 0:
            print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("5. Layer-Specific Queries")
    print("=" * 60)
    
    for layer in [MemoryLayer.WORKING, MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM]:
        results = memory_service.read_memory(
            query="user information",
            query_embedding=simple_embedding("user information"),
            limit=3,
            layer=layer,
        )
        print(f"✓ {layer.value.upper()} layer ({len(results)} results):")
        for entry in results:
            print(f"  - {entry.content[:60]}...")
        print()
    
    print("=" * 60)
    print("6. Memory Decay Simulation")
    print("=" * 60)
    
    print(f"✓ Before decay: {len(memory_service.memory_store)} memories")
    
    decay_stats = consolidator.apply_decay(
        memory_service.memory_store,
        memory_service.layer_indices,
        decay_rate=0.95,
    )
    
    print(f"✓ After decay: {len(memory_service.memory_store)} memories")
    print(f"✓ Forgotten: {decay_stats['forgotten']} memories\n")
    
    print("=" * 60)
    print("7. Batch Query Processing")
    print("=" * 60)
    
    batch_queries = [
        ("user preferences", simple_embedding("user preferences")),
        ("recent conversations", simple_embedding("recent conversations")),
        ("technical skills", simple_embedding("technical skills")),
    ]
    
    batch_results = query_engine.batch_query(
        batch_queries,
        memory_service.memory_store,
        memory_service.layer_indices,
        limit=3,
    )
    
    for (query, _), results in zip(batch_queries, batch_results):
        print(f"✓ Query: '{query}' - {len(results)} results")
    print()
    
    print("=" * 60)
    print("8. Final Memory Statistics")
    print("=" * 60)
    
    stats = memory_service.get_memory_stats()
    print(f"✓ Final Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Advanced example completed successfully!")


if __name__ == "__main__":
    main()
