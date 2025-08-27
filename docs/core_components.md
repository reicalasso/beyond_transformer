# Core Components of Neural State Machine Architecture

This document defines the core components of the Neural State Machine (NSM) architecture and how they relate to and integrate with existing architectures like Transformers, SSM (Mamba), NTM, and RNNs.

## 1. Core NSM Components

### 1.1 TokenToStateRouter

**Role**: Routes input tokens to appropriate state nodes based on learned routing mechanisms.

**Key Features**:
- Multi-head attention-based routing mechanism
- Learnable routing weights that determine which tokens should influence which states
- Dimension projection from token space to state space
- Soft routing allowing tokens to influence multiple states

**Integration with Existing Architectures**:
- Similar to attention mechanisms in Transformers but with explicit state nodes
- Complements SSM models by providing explicit routing instead of implicit state evolution
- Extends RNNs by allowing parallel routing instead of sequential processing
- Enhances NTM by providing learned routing instead of content-based addressing

**Implementation Details**:
```
class TokenToStateRouter(nn.Module):
    def __init__(self, token_dim, state_dim, num_states, num_heads=4):
        # Linear projection for routing
        self.router = nn.Linear(token_dim, num_states * num_heads)
        # Output projection to state dimension
        self.output_projection = nn.Linear(state_dim, state_dim)
    
    def forward(self, tokens, states):
        # Compute routing weights
        routing_logits = self.router(tokens)
        routing_weights = F.softmax(routing_logits, dim=-1)
        # Route tokens to states
        routed_tokens = torch.bmm(routing_weights.transpose(1, 2), tokens)
        return routed_tokens, routing_weights
```

### 1.2 StateManager

**Role**: Manages state nodes with dynamic allocation and pruning capabilities.

**Key Features**:
- Dynamic state allocation based on task complexity
- State pruning based on importance scores
- Learnable importance scores for each state node
- Active state tracking for efficient computation

**Integration with Existing Architectures**:
- Extends NTM by providing dynamic memory allocation
- Complements SSM by managing explicit state representations
- Enhances RNNs by allowing variable number of state nodes
- Provides interpretability that Transformers lack

**Implementation Details**:
```
class StateManager(nn.Module):
    def __init__(self, state_dim, max_states=64, initial_states=None, prune_threshold=0.1):
        # State nodes as learnable parameters
        self.states = nn.Parameter(torch.randn(max_states, state_dim))
        # Learnable importance scores
        self.importance_scores = nn.Parameter(torch.ones(max_states))
        # Active state tracking
        self.active_mask = torch.ones(max_states, dtype=torch.bool)
    
    def forward(self):
        # Return only active states
        return self.states[self.active_mask]
    
    def allocate_states(self, num_states):
        # Allocate additional state nodes
        pass
    
    def prune_states(self):
        # Prune low importance states
        pass
```

### 1.3 StatePropagator

**Role**: Controls state updates using gating mechanisms and enables state-to-state communication.

**Key Features**:
- LSTM/GRU-style gated updates for controlling state evolution
- State-to-state communication using MultiHeadAttention
- Support for both single and multiple state updates
- Residual connections and layer normalization

**Integration with Existing Architectures**:
- Combines RNN gating mechanisms with attention-based communication
- Extends SSM by providing explicit state updates
- Enhances NTM with learned state transitions
- Complements Transformers with persistent state representations

**Implementation Details**:
```
class StatePropagator(nn.Module):
    def __init__(self, state_dim, gate_type='gru', num_heads=4, enable_communication=True):
        # Initialize gating mechanisms (GRU or LSTM style)
        self._init_gates()
        # Initialize communication mechanism
        if enable_communication:
            self.attention = nn.MultiheadAttention(
                embed_dim=state_dim, num_heads=num_heads, batch_first=True)
    
    def forward(self, prev_state, new_input):
        # Apply gated update
        updated_state = self._gated_update(prev_state, new_input)
        # Apply state-to-state communication if enabled
        if self.enable_communication:
            updated_state = self._apply_communication(updated_state)
        return updated_state
```

### 1.4 HybridAttention

**Role**: Combines token-to-state routing with content-based attention.

**Key Features**:
- Attention from states to tokens for information gathering
- Multi-head attention mechanism
- Residual connections and layer normalization
- Compatibility with various attention patterns

**Integration with Existing Architectures**:
- Extends Transformer attention to explicit state representations
- Complements SSM by providing attention-based information routing
- Enhances NTM with content-based addressing mechanisms
- Provides structured attention that RNNs lack

**Implementation Details**:
```
class HybridAttention(nn.Module):
    def __init__(self, state_dim, token_dim, num_heads=4):
        # Linear projections
        self.state_projection = nn.Linear(state_dim, state_dim)
        self.token_projection = nn.Linear(token_dim, state_dim)
        # Attention scaling
        self.scale = (state_dim // num_heads) ** -0.5
    
    def forward(self, states, tokens):
        # Compute attention scores between states and tokens
        attention_scores = torch.matmul(states, tokens.transpose(-2, -1))
        attention_scores = attention_scores * self.scale
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply attention to tokens
        attended_tokens = torch.matmul(attention_weights, tokens)
        return attended_tokens
```

## 2. Integration with Existing Architectures

### 2.1 Transformer Integration

**Role**: Global context modeling
**Integration Approach**: 
- NSM can incorporate Transformer layers for global context when needed
- Hybrid attention mechanism can be extended with standard self-attention
- Token-to-state routing can complement attention patterns

**Source**: PyTorch, HuggingFace

### 2.2 SSM (Mamba) Integration

**Role**: Efficient long-term memory processing
**Integration Approach**:
- StatePropagator can be enhanced with SSM mechanisms for state evolution
- StateManager can manage Mamba state representations
- HybridAttention can route information to SSM states

**Source**: Mamba repository

### 2.3 NTM Integration

**Role**: External memory + read/write operations
**Integration Approach**:
- StateManager can be extended to include external memory matrices
- TokenToStateRouter can implement NTM-style addressing mechanisms
- StatePropagator can handle memory read/write operations

**Source**: NTM libraries (PyTorch implementations)

### 2.4 RNN (LSTM/GRU) Integration

**Role**: Short-term context modeling
**Integration Approach**:
- StatePropagator already implements LSTM/GRU-style gating
- TokenToStateRouter can provide parallel processing instead of sequential
- StateManager can dynamically adjust the number of "RNN cells"

**Source**: PyTorch

## 3. Component Interaction Flow

```
Input Tokens → TokenToStateRouter → HybridAttention → StatePropagator → Updated States
                     ↑                    ↑              ↑           ↑
               StateManager          StateManager   StateManager   StateManager
```

1. **TokenToStateRouter**: Routes input tokens to relevant states based on learned attention patterns
2. **StateManager**: Provides current active states and manages dynamic allocation/pruning
3. **HybridAttention**: Computes attention between states and routed tokens to gather information
4. **StatePropagator**: Updates state representations using gated mechanisms and enables state-to-state communication
5. **StateManager**: Updates importance scores and manages state lifecycle

## 4. Key Advantages of NSM Components

### 4.1 Explicit State Management
- Unlike Transformers that compute attention weights dynamically, NSM has explicit state nodes
- StateManager enables dynamic allocation and pruning based on task complexity
- Provides better interpretability through explicit state representations

### 4.2 Efficient Computation
- TokenToStateRouter enables sparse computation by routing tokens to relevant states
- StatePropagator uses gated updates for efficient state evolution
- HybridAttention focuses computation on relevant token-state pairs

### 4.3 Flexible Integration
- Components can be integrated with existing architectures
- Modular design allows for component replacement or enhancement
- Supports both sequential and parallel processing patterns

## 5. Implementation Status

### 5.1 Implemented Components
- ✅ TokenToStateRouter (in `src/nsm/components.py`)
- ✅ StateManager (in `src/nsm/modules/state_manager.py`)
- ✅ StatePropagator (in `src/nsm/modules/state_propagator.py`)
- ✅ HybridAttention (in `src/nsm/layers.py`)

### 5.2 Integration Points
- ⬜ Transformer integration (planned)
- ⬜ SSM (Mamba) integration (planned)
- ⬜ NTM integration (planned)
- ⬜ RNN integration (partially implemented through StatePropagator)

## 6. Future Enhancements

### 6.1 Advanced Routing Mechanisms
- Meta-learning approaches for routing
- Hierarchical state structures
- Adaptive routing based on task requirements

### 6.2 Enhanced State Management
- Hierarchical memory allocation
- Task-specific state initialization
- Cross-task state transfer

### 6.3 Improved Communication Patterns
- Graph-based state communication
- Temporal state evolution tracking
- Multi-scale state representations