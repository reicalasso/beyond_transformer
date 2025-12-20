"""
PULSE MaaS Example

Demonstrates core memory operations with the Memory-as-a-Service API.
"""

import torch
from pulse.maas import MemoryService, MemoryLayer, MemoryAPI


def simple_embedding(text: str, dim: int = 768) -> torch.Tensor:
    """Simple embedding function for demonstration."""
    hash_val = hash(text)
    torch.manual_seed(abs(hash_val) % (2**31))
    return torch.randn(dim)


def main():
    print("PULSE MaaS Example\n")
    
    memory_service = MemoryService(hidden_size=768)
    api = MemoryAPI(memory_service, embedding_fn=simple_embedding)
    
    print("Writing memories...")
    write_request = {
        "content": "User prefers Python for development",
        "type": "preference",
        "scope": "private",
        "layer": "long-term",
        "importance": 0.9,
    }
    response = api.write_memory(write_request)
    print(f"Written: {response['memory_id']}")
    
    write_request2 = {
        "content": "User is learning machine learning",
        "type": "fact",
        "scope": "private",
        "layer": "short-term",
    }
    api.write_memory(write_request2)
    
    print("\nQuerying memories...")
    read_request = {
        "query": "what does user prefer",
        "limit": 5,
    }
    response = api.read_memory(read_request)
    print(f"Found {response['count']} results:")
    for result in response["results"]:
        print(f"  - {result['content']} (Layer: {result['layer']})")
    
    print("\nMemory statistics:")
    stats = api.get_stats()
    for key, value in stats["stats"].items():
        print(f"  {key}: {value}")
    
    print("\nConsolidating memories...")
    response = api.consolidate_memories()
    print(f"Consolidated: {response['stats']['promoted']} promoted")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
