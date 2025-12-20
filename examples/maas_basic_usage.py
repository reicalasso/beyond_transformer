"""
PULSE + MaaS Basic Usage Example

Demonstrates basic memory operations with the Memory-as-a-Service API.
"""

import torch
from pulse.maas import MemoryService, MemoryLayer, MemoryScope, MemoryAPI


def simple_embedding(text: str, dim: int = 768) -> torch.Tensor:
    """Simple embedding function for demonstration."""
    hash_val = hash(text)
    torch.manual_seed(abs(hash_val) % (2**31))
    return torch.randn(dim)


def main():
    print("🔥 PULSE + MaaS Basic Usage Example\n")
    
    memory_service = MemoryService(hidden_size=768)
    api = MemoryAPI(memory_service, embedding_fn=simple_embedding)
    
    print("=" * 60)
    print("1. Writing Memories")
    print("=" * 60)
    
    write_request = {
        "content": "User prefers Rust for systems programming",
        "type": "preference",
        "scope": "private",
        "layer": "long-term",
        "importance": 0.9,
        "metadata": {"category": "programming", "language": "rust"}
    }
    
    response = api.write_memory(write_request)
    print(f"✓ Write response: {response}")
    memory_id = response["memory_id"]
    
    write_request2 = {
        "content": "User is learning machine learning with PyTorch",
        "type": "fact",
        "scope": "private",
        "layer": "short-term",
    }
    response2 = api.write_memory(write_request2)
    print(f"✓ Write response 2: {response2}\n")
    
    write_request3 = {
        "content": "User asked about async programming yesterday",
        "type": "conversation",
        "scope": "private",
        "layer": "working",
    }
    response3 = api.write_memory(write_request3)
    print(f"✓ Write response 3: {response3}\n")
    
    print("=" * 60)
    print("2. Reading Memories")
    print("=" * 60)
    
    read_request = {
        "query": "what programming language does user like",
        "limit": 3,
        "type": "preference",
    }
    
    response = api.read_memory(read_request)
    print(f"✓ Read response:")
    print(f"  Found {response['count']} results")
    for result in response["results"]:
        print(f"  - {result['content']}")
        print(f"    Layer: {result['layer']}, Importance: {result['importance']:.2f}\n")
    
    print("=" * 60)
    print("3. Updating Memory")
    print("=" * 60)
    
    update_request = {
        "memory_id": memory_id,
        "importance": 0.95,
        "metadata": {"updated": True, "priority": "high"}
    }
    
    response = api.update_memory(update_request)
    print(f"✓ Update response: {response}\n")
    
    print("=" * 60)
    print("4. Memory Statistics")
    print("=" * 60)
    
    response = api.get_stats()
    print(f"✓ Memory stats:")
    for key, value in response["stats"].items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("5. Memory Consolidation")
    print("=" * 60)
    
    response = api.consolidate_memories()
    print(f"✓ Consolidation stats:")
    for key, value in response["stats"].items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("6. Querying Different Layers")
    print("=" * 60)
    
    for layer in ["working", "short-term", "long-term"]:
        read_request = {
            "query": "user information",
            "limit": 5,
            "layer": layer,
        }
        response = api.read_memory(read_request)
        print(f"✓ {layer.upper()} layer: {response['count']} results")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
