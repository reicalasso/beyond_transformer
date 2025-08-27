# Forward/Backward Flow and Variable Interactions in Neural State Machine

This document defines the forward and backward computational flow in the Neural State Machine architecture and details how memory, attention, and state variables interact.

## 1. Forward Pass Flow

### 1.1 Overall Forward Flow

```
Input Sequence → Token Processing → State Initialization → 
Processing Loop (for each time step):
  ├── Token-to-State Routing
  ├── Hybrid Attention
  ├── State Propagation
  ├── State Management
  └── State Communication
→ Output Generation
```

### 1.2 Detailed Forward Pass Steps

#### Step 1: Input Processing
```
Input: [batch_size, sequence_length, token_dimension]
Process: Tokenization, embedding, positional encoding
Output: [batch_size, sequence_length, embedded_dimension]
```

#### Step 2: State Initialization
```
Input: None (or previous states for streaming)
Process: 
  - Initialize state nodes from StateManager
  - Set initial state values
Output: 
  - Initial states: [batch_size, num_states, state_dimension]
  - State metadata (importance scores, active mask)
```

#### Step 3: Processing Loop (for each token/segment)

##### 3.1 Token-to-State Routing
```
Inputs:
  - Tokens: [batch_size, segment_length, token_dimension]
  - Current States: [batch_size, num_states, state_dimension]

Process:
  1. Compute routing logits: 
     logits = linear_router(tokens)  # [batch_size, segment_length, num_states * num_heads]
  2. Reshape and apply softmax:
     routing_weights = softmax(logits.reshape(...))  # [batch_size, segment_length, num_states]
  3. Route tokens to states:
     routed_tokens = bmm(routing_weights.transpose(), tokens)  # [batch_size, num_states, token_dimension]
  4. Project to state dimension:
     attended_tokens = projection(routed_tokens)  # [batch_size, num_states, state_dimension]

Outputs:
  - Routed/attended tokens: [batch_size, num_states, state_dimension]
  - Routing weights: [batch_size, segment_length, num_states]
```

##### 3.2 Hybrid Attention
```
Inputs:
  - States: [batch_size, num_states, state_dimension]
  - Tokens: [batch_size, segment_length, token_dimension]

Process:
  1. Linear projections:
     states_proj = state_projection(states)  # [batch_size, num_states, state_dimension]
     tokens_proj = token_projection(tokens)  # [batch_size, segment_length, state_dimension]
  2. Reshape for multi-head attention:
     states_proj = states_proj.view(batch, num_states, num_heads, head_dim).transpose(1, 2)
     tokens_proj = tokens_proj.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
  3. Compute attention scores:
     attention_scores = matmul(states_proj, tokens_proj.transpose(-2, -1)) * scale
  4. Apply attention weights:
     attention_weights = softmax(attention_scores)
     attended_tokens = matmul(attention_weights, tokens_proj)
  5. Reshape and project output:
     attended_tokens = attended_tokens.transpose(1, 2).reshape(batch, num_states, state_dimension)
     attended_tokens = output_projection(attended_tokens)

Outputs:
  - Attended tokens: [batch_size, num_states, state_dimension]
  - Attention weights: [batch_size, num_heads, num_states, segment_length]
```

##### 3.3 State Propagation
```
Inputs:
  - Previous states: [batch_size, num_states, state_dimension]
  - Attended tokens: [batch_size, num_states, state_dimension]

Process (GRU-style):
  1. Concatenate state and input:
     combined = cat([prev_states, attended_tokens], dim=-1)
  2. Compute gates:
     reset_gate = sigmoid(reset_gate_linear(combined))
     update_gate = sigmoid(update_gate_linear(combined))
  3. Compute candidate state:
     reset_prev_states = reset_gate * prev_states
     candidate_combined = cat([reset_prev_states, attended_tokens], dim=-1)
     candidate_state = tanh(candidate_state_linear(candidate_combined))
  4. Update state:
     new_states = (1 - update_gate) * prev_states + update_gate * candidate_state

Process (Optional State-to-State Communication):
  1. Apply self-attention to states:
     updated_states, _ = multihead_attention(states, states, states)
  2. Add residual and normalize:
     updated_states = layer_norm(states + updated_states)

Outputs:
  - Updated states: [batch_size, num_states, state_dimension]
```

##### 3.4 State Management
```
Inputs:
  - Updated states: [batch_size, num_states, state_dimension]
  - Attention weights (optional)
  - Gradients (optional, for importance updates)

Process:
  1. Update importance scores (if needed):
     if attention_weights:
       importance_update = mean(attention_weights, dim=[0, 1])  # Average across batch and heads
       importance_scores = ema_update(importance_scores, importance_update)
  2. Dynamic allocation/pruning:
     if should_manage:
       allocate_states(num_to_allocate)
       prune_states()

Outputs:
  - State metadata updates (importance scores, active mask)
```

#### Step 4: Output Generation
```
Input: Final states [batch_size, num_states, state_dimension]
Process:
  1. Global pooling: 
     pooled_states = mean(states, dim=1)  # [batch_size, state_dimension]
     OR
     pooled_states = states.view(batch_size, -1)  # [batch_size, num_states * state_dimension]
  2. Output projection:
     output = linear_output(pooled_states)  # [batch_size, output_dimension]

Output: Final predictions [batch_size, output_dimension]
```

## 2. Backward Pass Flow

### 2.1 Gradient Flow Path

```
Loss → Output Layer → State Processing Loop → 
Token-to-State Router → Hybrid Attention → State Propagator →
State Manager (importance updates) → Input Processing
```

### 2.2 Detailed Backward Pass Steps

#### Step 1: Output Layer Backward
```
Input: Gradients from loss function
Process: 
  - Backpropagate through output projection
  - Backpropagate through pooling operation
Output: Gradients w.r.t. final states
```

#### Step 2: State Processing Loop Backward
```
Input: Gradients w.r.t. updated states
Process (reverse order):
  1. State-to-State Communication Backward:
     - Backpropagate through layer norm
     - Backpropagate through residual connection
     - Backpropagate through multi-head attention

  2. State Propagation Backward:
     - Backpropagate through state update formula
     - Backpropagate through candidate state computation
     - Backpropagate through gate computations

Output: Gradients w.r.t. attended tokens and previous states
```

#### Step 3: Hybrid Attention Backward
```
Input: Gradients w.r.t. attended tokens
Process:
  - Backpropagate through output projection
  - Backpropagate through attention application
  - Backpropagate through softmax
  - Backpropagate through attention score computation
  - Backpropagate through linear projections

Output: Gradients w.r.t. states and tokens
```

#### Step 4: Token-to-State Router Backward
```
Input: Gradients w.r.t. routed tokens
Process:
  - Backpropagate through dimension projection
  - Backpropagate through routing operation
  - Backpropagate through softmax
  - Backpropagate through linear router

Output: Gradients w.r.t. input tokens
```

#### Step 5: State Manager Backward
```
Input: Gradients w.r.t. state parameters
Process:
  - Accumulate gradients for importance scores
  - Accumulate gradients for state node parameters
  - Update importance scores based on attention/gradient information

Output: Updated parameter gradients
```

## 3. Memory, Attention, and State Variable Interactions

### 3.1 Memory Variables

#### 3.1.1 State Nodes
```
Variable: State Nodes
Shape: [max_states, state_dimension]
Type: Learnable Parameters
Persistence: Long-term (across sequences)
Update Mechanism: Gated updates + gradient descent
Access Pattern: Selective (via routing/attention)
```

#### 3.1.2 Importance Scores
```
Variable: Importance Scores
Shape: [max_states]
Type: Learnable Parameters
Persistence: Long-term (across sequences)
Update Mechanism: 
  - Gradient-based during training
  - Exponential moving average based on attention weights
Access Pattern: Used for dynamic allocation/pruning
```

#### 3.1.3 Active State Mask
```
Variable: Active State Mask
Shape: [max_states]
Type: Buffer (not learnable)
Persistence: Short-term (can change during processing)
Update Mechanism: Dynamic allocation/pruning operations
Access Pattern: Determines which states are active
```

### 3.2 Attention Variables

#### 3.2.1 Routing Weights
```
Variable: Routing Weights
Shape: [batch_size, segment_length, num_states]
Type: Computed during forward pass
Persistence: Per forward pass
Update Mechanism: Computed from linear projection + softmax
Access Pattern: Determines token-to-state mapping
```

#### 3.2.2 Attention Weights
```
Variable: Attention Weights
Shape: [batch_size, num_heads, num_states, segment_length]
Type: Computed during forward pass
Persistence: Per forward pass
Update Mechanism: Computed from state-token attention + softmax
Access Pattern: Determines information flow from tokens to states
```

### 3.3 State Variables

#### 3.3.1 Current States
```
Variable: Current States
Shape: [batch_size, num_active_states, state_dimension]
Type: Computed during forward pass
Persistence: Per sequence/segment
Update Mechanism: Gated updates from StatePropagator
Access Pattern: 
  - Input to attention mechanisms
  - Input to state propagation
  - Output from state propagation
```

#### 3.3.2 State Gates
```
Variable: State Gates (Reset, Update, Forget, etc.)
Shape: [batch_size, num_states, state_dimension]
Type: Computed during forward pass
Persistence: Per state update
Update Mechanism: Linear projection + activation
Access Pattern: Control state update dynamics
```

### 3.4 Interaction Patterns

#### 3.4.1 Memory-State Interaction
```
Direction: Bidirectional
Mechanism:
  1. States read from memory nodes (parameters)
  2. State updates modify memory nodes (via gradients)
  3. Dynamic allocation/pruning changes active memory subset
Frequency: 
  - Read: Every state update
  - Write: Every parameter update
  - Management: Based on importance scores
```

#### 3.4.2 Attention-State Interaction
```
Direction: Bidirectional
Mechanism:
  1. States attend to tokens via attention weights
  2. Attention weights influence state updates
  3. Routing weights determine which tokens affect which states
Frequency:
  - Attention computation: Per segment
  - Weight updates: Per training step
```

#### 3.4.3 State-State Interaction
```
Direction: Bidirectional
Mechanism:
  1. States communicate via self-attention
  2. Collective state evolution through communication
  3. Residual connections maintain state identity
Frequency:
  - Communication: Per state update (if enabled)
  - Update: Every processing step
```

## 4. Data Flow Characteristics

### 4.1 Computational Complexity
```
Token-to-State Router: O(batch * segment_length * num_states * token_dim)
Hybrid Attention: O(batch * num_heads * num_states * segment_length * head_dim)
State Propagation: O(batch * num_states * state_dim²)
State Communication: O(batch * num_heads * num_states² * head_dim)
State Management: O(max_states)
```

### 4.2 Memory Requirements
```
State Storage: O(max_states * state_dim)
Attention Cache: O(batch * num_heads * num_states * segment_length)
Intermediate Activations: O(batch * num_states * state_dim)
Gradients: O(same as forward pass variables)
```

### 4.3 Parallelization Opportunities
```
Across Batches: Full parallelization
Across States: Parallel state updates
Across Tokens: Parallel routing within segments
Across Heads: Parallel multi-head attention
```

## 5. Variable Lifecycle

### 5.1 Long-term Variables (Persist across sequences)
- State nodes (parameters)
- Importance scores (parameters)
- Active state mask (buffer)

### 5.2 Medium-term Variables (Persist across segments)
- Current states (computed)
- State metadata

### 5.3 Short-term Variables (Per computation step)
- Routing weights
- Attention weights
- Intermediate computations
- Gradients

## 6. Gradient Flow Analysis

### 6.1 Critical Paths
```
Input Tokens → Token-to-State Router → Hybrid Attention → 
State Propagator → Output Layer → Loss
```

### 6.2 Gradient Information Routes
```
Loss Gradients → Output Layer → 
State Updates → Attention Mechanisms → 
Routing Mechanisms → Input Embeddings
```

### 6.3 Gradient Modulation Points
```
1. Attention weights modulate token influence on states
2. Routing weights control which tokens affect which states
3. Gate values control state update dynamics
4. Importance scores influence memory management
```

## 7. Implementation Considerations

### 7.1 Numerical Stability
```
- Softmax with numerical stabilization
- Gradient clipping for RNN-style updates
- Proper initialization of state nodes
- Stable sigmoid/tanh implementations
```

### 7.2 Memory Management
```
- Efficient state slicing based on active mask
- Gradient checkpointing for long sequences
- In-place operations where possible
- Memory pooling for intermediate variables
```

### 7.3 Computational Efficiency
```
- Sparse attention for long sequences
- Efficient matrix multiplication libraries
- Parallel processing of independent components
- Caching of repeated computations
```

## 8. Debugging and Monitoring Points

### 8.1 Key Variables to Monitor
```
- State norm evolution (detect exploding/vanishing states)
- Attention weight sparsity (routing efficiency)
- Importance score distribution (memory utilization)
- Gate activation patterns (update dynamics)
```

### 8.2 Diagnostic Tools
```
- Attention visualization
- State trajectory analysis
- Routing pattern analysis
- Memory allocation/pruning logs
```

This comprehensive specification defines how memory, attention, and state variables interact in the Neural State Machine architecture, providing a clear understanding of both forward and backward computational flows.