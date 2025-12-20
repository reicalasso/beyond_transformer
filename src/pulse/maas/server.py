"""
PULSE MaaS Server

Flask-based REST API server for Memory-as-a-Service.
Provides HTTP endpoints for all memory operations.
"""

from typing import Optional
import torch

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask flask-cors")

from .memory_service import MemoryService
from .api import MemoryAPI
from .consolidation import MemoryConsolidator
from .query_engine import MemoryQueryEngine


class MaaSServer:
    """
    Memory-as-a-Service Server
    
    Provides REST API for PULSE memory operations.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        host: str = "0.0.0.0",
        port: int = 5000,
        embedding_fn: Optional[callable] = None,
    ):
        """
        Initialize MaaS server.
        
        Args:
            hidden_size: Embedding dimension
            host: Server host
            port: Server port
            embedding_fn: Custom embedding function
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required. Install with: pip install flask flask-cors")
        
        self.host = host
        self.port = port
        
        self.memory_service = MemoryService(hidden_size=hidden_size)
        self.api = MemoryAPI(self.memory_service, embedding_fn)
        self.consolidator = MemoryConsolidator()
        self.query_engine = MemoryQueryEngine(hidden_size=hidden_size)
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/pulse/memory/write', methods=['POST'])
        def write_memory():
            """Write memory endpoint."""
            data = request.get_json()
            response = self.api.write_memory(data)
            return jsonify(response)
        
        @self.app.route('/pulse/memory/read', methods=['POST'])
        def read_memory():
            """Read memory endpoint."""
            data = request.get_json()
            response = self.api.read_memory(data)
            return jsonify(response)
        
        @self.app.route('/pulse/memory/update', methods=['PUT'])
        def update_memory():
            """Update memory endpoint."""
            data = request.get_json()
            response = self.api.update_memory(data)
            return jsonify(response)
        
        @self.app.route('/pulse/memory/delete', methods=['DELETE'])
        def delete_memory():
            """Delete memory endpoint."""
            data = request.get_json()
            response = self.api.delete_memory(data)
            return jsonify(response)
        
        @self.app.route('/pulse/memory/consolidate', methods=['POST'])
        def consolidate():
            """Consolidate memories endpoint."""
            data = request.get_json() if request.data else {}
            response = self.api.consolidate_memories(data)
            return jsonify(response)
        
        @self.app.route('/pulse/memory/stats', methods=['GET'])
        def get_stats():
            """Get statistics endpoint."""
            response = self.api.get_stats()
            return jsonify(response)
        
        @self.app.route('/pulse/memory/query/advanced', methods=['POST'])
        def advanced_query():
            """Advanced query with dynamic routing."""
            data = request.get_json()
            query = data.get("query")
            
            if not query:
                return jsonify({"success": False, "error": "Query is required"})
            
            query_embedding = self.api.embedding_fn(query)
            
            active_layers = self.query_engine.dynamic_route(
                query_embedding,
                self.memory_service.memory_store,
                self.memory_service.layer_indices,
            )
            
            results = self.query_engine.query(
                query_text=query,
                query_embedding=query_embedding,
                memory_store=self.memory_service.memory_store,
                layer_indices=self.memory_service.layer_indices,
                limit=data.get("limit", 10),
                layers=active_layers,
            )
            
            return jsonify({
                "success": True,
                "active_layers": [layer.value for layer in active_layers],
                "results": [entry.to_dict() for score, entry in results],
                "scores": [score for score, entry in results],
                "count": len(results),
            })
        
        @self.app.route('/pulse/health', methods=['GET'])
        def health():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "service": "PULSE MaaS",
                "version": "1.0.0",
            })
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Index endpoint with API documentation."""
            return jsonify({
                "service": "PULSE Memory-as-a-Service",
                "version": "1.0.0",
                "endpoints": {
                    "POST /pulse/memory/write": "Write new memory",
                    "POST /pulse/memory/read": "Query memories",
                    "PUT /pulse/memory/update": "Update memory",
                    "DELETE /pulse/memory/delete": "Delete memory",
                    "POST /pulse/memory/consolidate": "Trigger consolidation",
                    "GET /pulse/memory/stats": "Get statistics",
                    "POST /pulse/memory/query/advanced": "Advanced query with routing",
                    "GET /pulse/health": "Health check",
                },
                "features": [
                    "Hierarchical memory (working, short-term, long-term)",
                    "Automatic consolidation and decay",
                    "Dynamic routing to relevant memories",
                    "Semantic search and retrieval",
                    "Human-like memory patterns",
                ],
            })
    
    def run(self, debug: bool = False):
        """
        Run the MaaS server.
        
        Args:
            debug: Enable debug mode
        """
        print(f"🔥 PULSE MaaS Server starting on {self.host}:{self.port}")
        print(f"📊 Memory stats: {self.memory_service.get_memory_stats()}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_server(
    hidden_size: int = 768,
    host: str = "0.0.0.0",
    port: int = 5000,
    embedding_fn: Optional[callable] = None,
) -> MaaSServer:
    """
    Create a MaaS server instance.
    
    Args:
        hidden_size: Embedding dimension
        host: Server host
        port: Server port
        embedding_fn: Custom embedding function
        
    Returns:
        MaaSServer instance
    """
    return MaaSServer(
        hidden_size=hidden_size,
        host=host,
        port=port,
        embedding_fn=embedding_fn,
    )


if __name__ == "__main__":
    server = create_server()
    server.run(debug=True)
