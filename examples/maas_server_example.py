"""
PULSE + MaaS Server Example

Demonstrates running the MaaS server and making HTTP requests.
"""

import time
import requests
import json


def test_server(base_url: str = "http://localhost:5000"):
    """Test the MaaS server with various requests."""
    
    print("🔥 PULSE + MaaS Server Test\n")
    
    print("=" * 60)
    print("1. Health Check")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/pulse/health")
    print(f"✓ Health: {response.json()}\n")
    
    print("=" * 60)
    print("2. Write Memories")
    print("=" * 60)
    
    memories = [
        {
            "content": "User prefers Python for data science",
            "type": "preference",
            "scope": "private",
            "layer": "long-term",
            "importance": 0.9,
        },
        {
            "content": "User is working on a neural network project",
            "type": "fact",
            "scope": "private",
            "layer": "short-term",
        },
        {
            "content": "User asked about transformers 5 minutes ago",
            "type": "conversation",
            "scope": "private",
            "layer": "working",
        },
    ]
    
    memory_ids = []
    for mem in memories:
        response = requests.post(
            f"{base_url}/pulse/memory/write",
            json=mem,
            headers={"Content-Type": "application/json"}
        )
        result = response.json()
        print(f"✓ Written: {mem['content'][:50]}...")
        print(f"  ID: {result.get('memory_id')}, Layer: {result.get('layer')}")
        memory_ids.append(result.get('memory_id'))
    print()
    
    print("=" * 60)
    print("3. Query Memories")
    print("=" * 60)
    
    query = {
        "query": "what is user working on",
        "limit": 5,
    }
    
    response = requests.post(
        f"{base_url}/pulse/memory/read",
        json=query,
        headers={"Content-Type": "application/json"}
    )
    result = response.json()
    print(f"✓ Found {result.get('count')} results:")
    for mem in result.get('results', []):
        print(f"  - {mem['content']}")
        print(f"    Layer: {mem['layer']}, Importance: {mem['importance']:.2f}")
    print()
    
    print("=" * 60)
    print("4. Advanced Query with Dynamic Routing")
    print("=" * 60)
    
    query = {
        "query": "user preferences and current work",
        "limit": 10,
    }
    
    response = requests.post(
        f"{base_url}/pulse/memory/query/advanced",
        json=query,
        headers={"Content-Type": "application/json"}
    )
    result = response.json()
    print(f"✓ Active layers: {result.get('active_layers')}")
    print(f"✓ Found {result.get('count')} results")
    for i, (mem, score) in enumerate(zip(result.get('results', []), result.get('scores', []))):
        print(f"  {i+1}. {mem['content'][:60]}... (score: {score:.3f})")
    print()
    
    print("=" * 60)
    print("5. Memory Statistics")
    print("=" * 60)
    
    response = requests.get(f"{base_url}/pulse/memory/stats")
    stats = response.json().get('stats', {})
    print(f"✓ Memory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("=" * 60)
    print("6. Update Memory")
    print("=" * 60)
    
    if memory_ids:
        update = {
            "memory_id": memory_ids[0],
            "importance": 0.95,
            "metadata": {"priority": "high", "updated": True}
        }
        
        response = requests.put(
            f"{base_url}/pulse/memory/update",
            json=update,
            headers={"Content-Type": "application/json"}
        )
        print(f"✓ Update response: {response.json()}\n")
    
    print("=" * 60)
    print("7. Consolidation")
    print("=" * 60)
    
    response = requests.post(
        f"{base_url}/pulse/memory/consolidate",
        json={},
        headers={"Content-Type": "application/json"}
    )
    result = response.json()
    print(f"✓ Consolidation stats:")
    for key, value in result.get('stats', {}).items():
        print(f"  {key}: {value}")
    print()
    
    print("✅ All tests completed successfully!")


def start_server_instructions():
    """Print instructions for starting the server."""
    print("=" * 60)
    print("To start the MaaS server, run:")
    print("=" * 60)
    print()
    print("  python -m pulse.maas.server")
    print()
    print("Or in Python:")
    print()
    print("  from pulse.maas.server import create_server")
    print("  server = create_server(port=5000)")
    print("  server.run(debug=True)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:5000"
        test_server(base_url)
    else:
        start_server_instructions()
        print("\nRun with 'python maas_server_example.py test' to test the server")
