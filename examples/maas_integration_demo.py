"""
PULSE + MaaS Integration Demo

Complete demonstration of the PULSE + MaaS system showing:
- Memory lifecycle (write → consolidate → query → forget)
- All three memory layers in action
- Dynamic routing
- Real-world use case simulation
"""

import torch
import time
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


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def simulate_chatbot_session():
    """Simulate a chatbot session with memory management."""
    
    print("🔥 PULSE + MaaS Integration Demo")
    print("Simulating an AI chatbot with hierarchical memory\n")
    
    memory_service = MemoryService(
        hidden_size=768,
        working_slots=32,
        short_term_slots=128,
        long_term_slots=512,
    )
    
    consolidator = MemoryConsolidator(
        working_ttl=5.0,
        short_term_ttl=15.0,
    )
    
    query_engine = MemoryQueryEngine(hidden_size=768)
    
    print_section("Session 1: User Onboarding")
    
    onboarding_memories = [
        ("User's name is Alice", "fact", 0.95, MemoryLayer.LONG_TERM),
        ("Alice prefers Python for data science", "preference", 0.9, MemoryLayer.LONG_TERM),
        ("Alice is interested in machine learning", "interest", 0.85, MemoryLayer.LONG_TERM),
        ("Alice works at a tech startup", "fact", 0.8, MemoryLayer.LONG_TERM),
    ]
    
    for content, mem_type, importance, layer in onboarding_memories:
        memory_id = memory_service.write_memory(
            content=content,
            embedding=simple_embedding(content),
            memory_type=mem_type,
            scope=MemoryScope.PRIVATE,
            layer=layer,
            importance=importance,
        )
        print(f"✓ Stored: {content}")
    
    stats = memory_service.get_memory_stats()
    print(f"\n📊 Stats: {stats['total_memories']} memories, "
          f"{stats['long_term']} in long-term")
    
    print_section("Session 2: Active Conversation")
    
    conversation_memories = [
        ("Alice asked about neural networks", "conversation", 0.6, MemoryLayer.WORKING),
        ("Discussed backpropagation algorithm", "conversation", 0.65, MemoryLayer.WORKING),
        ("Alice mentioned working on image classification", "fact", 0.75, MemoryLayer.SHORT_TERM),
        ("Recommended PyTorch tutorials", "conversation", 0.5, MemoryLayer.WORKING),
    ]
    
    for content, mem_type, importance, layer in conversation_memories:
        memory_id = memory_service.write_memory(
            content=content,
            embedding=simple_embedding(content),
            memory_type=mem_type,
            scope=MemoryScope.PRIVATE,
            layer=layer,
            importance=importance,
        )
        print(f"✓ Stored: {content}")
    
    stats = memory_service.get_memory_stats()
    print(f"\n📊 Stats: Working={stats['working']}, "
          f"Short-term={stats['short_term']}, Long-term={stats['long_term']}")
    
    print_section("Query 1: What does Alice like?")
    
    query = "what does Alice prefer and like"
    query_emb = simple_embedding(query)
    
    active_layers = query_engine.dynamic_route(
        query_emb,
        memory_service.memory_store,
        memory_service.layer_indices,
    )
    
    print(f"🎯 Dynamic Routing activated layers: {[l.value for l in active_layers]}")
    
    results = query_engine.query(
        query_text=query,
        query_embedding=query_emb,
        memory_store=memory_service.memory_store,
        layer_indices=memory_service.layer_indices,
        limit=5,
        layers=active_layers,
    )
    
    print(f"\n✓ Found {len(results)} relevant memories:\n")
    for i, (score, entry) in enumerate(results, 1):
        print(f"  {i}. {entry.content}")
        print(f"     Score: {score:.3f}, Layer: {entry.layer.value}, "
              f"Importance: {entry.importance:.2f}")
    
    print_section("Query 2: Recent conversation topics")
    
    query = "what did we discuss recently"
    query_emb = simple_embedding(query)
    
    results = memory_service.read_memory(
        query=query,
        query_embedding=query_emb,
        limit=5,
        layer=MemoryLayer.WORKING,
    )
    
    print(f"✓ Found {len(results)} recent conversations:\n")
    for entry in results:
        print(f"  - {entry.content}")
        print(f"    Access count: {entry.access_count}")
    
    print_section("Memory Consolidation")
    
    print("⏰ Waiting for memories to age...")
    time.sleep(2)
    
    print("\n🔄 Running consolidation...")
    consolidation_stats = consolidator.consolidate(
        memory_service.memory_store,
        memory_service.layer_indices,
    )
    
    print("\n✓ Consolidation results:")
    for key, value in consolidation_stats.items():
        if value > 0:
            print(f"  {key}: {value}")
    
    stats = memory_service.get_memory_stats()
    print(f"\n📊 After consolidation: Working={stats['working']}, "
          f"Short-term={stats['short_term']}, Long-term={stats['long_term']}")
    
    print_section("Session 3: New Conversation")
    
    new_memories = [
        ("Alice asked about transformers", "conversation", 0.7, MemoryLayer.WORKING),
        ("Explained attention mechanism", "conversation", 0.65, MemoryLayer.WORKING),
        ("Alice is now learning about BERT", "fact", 0.8, MemoryLayer.SHORT_TERM),
    ]
    
    for content, mem_type, importance, layer in new_memories:
        memory_id = memory_service.write_memory(
            content=content,
            embedding=simple_embedding(content),
            memory_type=mem_type,
            scope=MemoryScope.PRIVATE,
            layer=layer,
            importance=importance,
        )
        print(f"✓ Stored: {content}")
    
    print_section("Query 3: Alice's learning journey")
    
    query = "what is Alice learning and working on"
    query_emb = simple_embedding(query)
    
    results = query_engine.query(
        query_text=query,
        query_embedding=query_emb,
        memory_store=memory_service.memory_store,
        layer_indices=memory_service.layer_indices,
        limit=10,
    )
    
    print(f"✓ Complete learning journey ({len(results)} memories):\n")
    for i, (score, entry) in enumerate(results, 1):
        print(f"  {i}. {entry.content}")
        print(f"     Layer: {entry.layer.value}, Importance: {entry.importance:.2f}, "
              f"Accessed: {entry.access_count} times")
    
    print_section("Memory Decay & Forgetting")
    
    print("🧹 Applying memory decay...")
    decay_stats = consolidator.apply_decay(
        memory_service.memory_store,
        memory_service.layer_indices,
        decay_rate=0.9,
    )
    
    print(f"\n✓ Decay results:")
    print(f"  Forgotten: {decay_stats['forgotten']} low-importance memories")
    
    stats = memory_service.get_memory_stats()
    print(f"\n📊 Final stats: {stats['total_memories']} total memories")
    print(f"  Working: {stats['working']}")
    print(f"  Short-term: {stats['short_term']}")
    print(f"  Long-term: {stats['long_term']}")
    print(f"  Average importance: {stats['avg_importance']:.2f}")
    
    print_section("Summary")
    
    print("✅ Demo completed successfully!\n")
    print("Key features demonstrated:")
    print("  ✓ Hierarchical memory (working, short-term, long-term)")
    print("  ✓ Dynamic routing (smart layer selection)")
    print("  ✓ Automatic consolidation (time-based promotion)")
    print("  ✓ Memory decay and forgetting")
    print("  ✓ Semantic search and retrieval")
    print("  ✓ Access pattern tracking")
    print("\nThis is how PULSE + MaaS provides human-like memory for AI! 🔥")


if __name__ == "__main__":
    simulate_chatbot_session()
