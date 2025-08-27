"""
Debuggable Neural State Machine Components

This module implements debuggable versions of key NSM components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from nsm.utils.debugger import DebuggableNSMComponent


class DebuggableTokenToStateRouter(DebuggableNSMComponent):
    """
    Debuggable version of TokenToStateRouter.
    """
    
    def __init__(self, token_dim: int, state_dim: int, num_states: int, 
                 num_heads: int = 4, debug_mode: bool = False):
        """
        Initialize debuggable token-to-state router.
        
        Args:
            token_dim: Dimension of input tokens
            state_dim: Dimension of state vectors
            num_states: Number of state nodes
            num_heads: Number of routing heads
            debug_mode: Whether to enable debug mode
        """
        super().__init__("TokenToStateRouter", debug_mode)
        
        self.token_dim = token_dim
        self.state_dim = state_dim
        self.num_states = num_states
        self.num_heads = num_heads
        
        # Routing mechanism
        self.router = nn.Linear(token_dim, num_states * num_heads)
        self.head_dim = state_dim // num_heads
        
        # Ensure compatibility
        assert state_dim % num_heads == 0, "state_dim must be divisible by num_heads"
        
        # Output projection
        self.output_projection = nn.Linear(state_dim, state_dim)
    
    def forward(self, tokens: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to states with debugging.
        
        Args:
            tokens: Input tokens [batch_size, seq_len, token_dim]
            states: State vectors [batch_size, num_states, state_dim]
            
        Returns:
            Tuple of (routed_tokens, routing_weights)
        """
        batch_size, seq_len, token_dim = tokens.shape
        num_states = states.shape[1]
        
        if self.debug_mode:
            self.log_step("input_processing", {
                'tokens': tokens,
                'states': states
            }, {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'token_dim': token_dim,
                'num_states': num_states
            })
        
        # Compute routing weights
        routing_logits = self.router(tokens)  # [batch_size, seq_len, num_states * num_heads]
        routing_logits = routing_logits.view(batch_size, seq_len, self.num_heads, num_states)
        
        if self.debug_mode:
            self.log_step("routing_logits_computed", {
                'routing_logits': routing_logits
            })
        
        # Average across heads
        routing_weights = F.softmax(routing_logits.mean(dim=2), dim=-1)  # [batch_size, seq_len, num_states]
        
        if self.debug_mode:
            self.log_step("routing_weights_computed", {
                'routing_weights': routing_weights
            })
            self.log_attention_operation(
                'token_to_state', f'tokens_{seq_len}', f'states_{num_states}',
                routing_weights
            )
        
        # Route tokens to states
        # tokens: [batch_size, seq_len, token_dim]
        # routing_weights: [batch_size, seq_len, num_states]
        # Result: [batch_size, num_states, token_dim]
        routed_tokens = torch.bmm(routing_weights.transpose(1, 2), tokens)
        
        if self.debug_mode:
            self.log_step("tokens_routed", {
                'routed_tokens': routed_tokens
            })
        
        # Project to state dimension
        if token_dim != self.state_dim:
            if token_dim < self.state_dim:
                routed_tokens = F.pad(routed_tokens, (0, self.state_dim - token_dim))
            else:
                routed_tokens = routed_tokens[:, :, :self.state_dim]
        
        # Apply output projection
        routed_tokens = self.output_projection(routed_tokens)
        
        if self.debug_mode:
            self.log_step("output_projection_applied", {
                'final_routed_tokens': routed_tokens,
                'final_routing_weights': routing_weights
            })
        
        return routed_tokens, routing_weights


class DebuggableStateManager(DebuggableNSMComponent):
    """
    Debuggable version of StateManager.
    """
    
    def __init__(self, state_dim: int, max_states: int = 64, 
                 initial_states: Optional[int] = None, 
                 prune_threshold: float = 0.1, debug_mode: bool = False):
        """
        Initialize debuggable state manager.
        
        Args:
            state_dim: Dimension of each state vector
            max_states: Maximum number of state nodes
            initial_states: Initial number of active states
            prune_threshold: Threshold for pruning states (0-1)
            debug_mode: Whether to enable debug mode
        """
        super().__init__("StateManager", debug_mode)
        
        self.state_dim = state_dim
        self.max_states = max_states
        self.prune_threshold = prune_threshold
        
        # Initialize state nodes
        initial_states = initial_states or max_states
        self.states = nn.Parameter(torch.randn(max_states, state_dim))
        
        # Learnable importance scores for each state node
        self.importance_scores = nn.Parameter(torch.ones(max_states))
        
        # Track active states
        self.register_buffer('active_mask', torch.ones(max_states, dtype=torch.bool))
        self.active_mask[initial_states:].fill_(False)
        
        # Initialize parameters
        nn.init.normal_(self.states, mean=0.0, std=0.1)
        nn.init.uniform_(self.importance_scores, 0.5, 1.0)
    
    def forward(self) -> torch.Tensor:
        """
        Get current active states with debugging.
        
        Returns:
            Active state vectors [num_active_states, state_dim]
        """
        active_states = self.states[self.active_mask]
        
        if self.debug_mode:
            self.log_step("active_states_retrieved", {
                'active_states': active_states,
                'active_count': self.get_active_count(),
                'importance_scores': torch.sigmoid(self.importance_scores)
            })
        
        return active_states
    
    def get_importance_scores(self) -> torch.Tensor:
        """Get importance scores for all states."""
        return torch.sigmoid(self.importance_scores)
    
    def get_active_count(self) -> int:
        """Get number of currently active states."""
        return self.active_mask.sum().item()
    
    def prune_low_importance_states(self) -> int:
        """
        Prune states below importance threshold with debugging.
        
        Returns:
            Number of states pruned
        """
        with torch.no_grad():
            importance_scores = self.get_importance_scores()
            low_importance_mask = (importance_scores < self.prune_threshold) & self.active_mask
            
            # Don't prune if it would leave fewer than 1 state
            would_prune = low_importance_mask.sum().item()
            if self.get_active_count() - would_prune >= 1:
                # Avoid in-place operations that can cause gradient issues
                new_mask = self.active_mask.clone()
                new_mask[low_importance_mask] = False
                self.active_mask.copy_(new_mask)
                
                if self.debug_mode:
                    self.log_step("states_pruned", {
                        'pruned_count': would_prune,
                        'remaining_count': self.get_active_count(),
                        'pruned_indices': torch.where(low_importance_mask)[0].tolist()
                    })
                    self.log_memory_operation(
                        'prune', f'pruned_{would_prune}_states',
                        operation_info={
                            'threshold': self.prune_threshold,
                            'scores_before_pruning': importance_scores.tolist()
                        }
                    )
                
                return would_prune
            else:
                return 0
    
    def allocate_states(self, num_states: int) -> int:
        """
        Allocate additional state nodes with debugging.
        
        Args:
            num_states: Number of states to allocate
            
        Returns:
            Number of states actually allocated
        """
        with torch.no_grad():
            current_active = self.get_active_count()
            available_slots = self.max_states - current_active
            
            if available_slots <= 0:
                return 0
                
            num_to_allocate = min(num_states, available_slots)
            
            # Find inactive slots
            inactive_indices = torch.where(~self.active_mask)[0][:num_to_allocate]
            
            # Activate these slots (avoid in-place operations)
            new_mask = self.active_mask.clone()
            new_mask[inactive_indices] = True
            self.active_mask.copy_(new_mask)
            
            # Initialize new states
            nn.init.normal_(self.states[inactive_indices], mean=0.0, std=0.1)
            
            if self.debug_mode:
                self.log_step("states_allocated", {
                    'allocated_count': num_to_allocate,
                    'new_active_count': self.get_active_count(),
                    'allocated_indices': inactive_indices.tolist()
                })
                self.log_memory_operation(
                    'allocate', f'allocated_{num_to_allocate}_states',
                    operation_info={
                        'requested': num_states,
                        'available': available_slots
                    }
                )
            
            return num_to_allocate


class DebuggableStatePropagator(DebuggableNSMComponent):
    """
    Debuggable version of StatePropagator.
    """
    
    def __init__(self, state_dim: int, gate_type: str = 'gru', 
                 num_heads: int = 4, enable_communication: bool = True,
                 debug_mode: bool = False):
        """
        Initialize debuggable state propagator.
        
        Args:
            state_dim: Dimension of the state vectors
            gate_type: Type of gating mechanism ('lstm' or 'gru')
            num_heads: Number of attention heads for state-to-state communication
            enable_communication: Whether to enable state-to-state communication
            debug_mode: Whether to enable debug mode
        """
        super().__init__("StatePropagator", debug_mode)
        
        self.state_dim = state_dim
        self.gate_type = gate_type
        self.enable_communication = enable_communication
        
        # Initialize gating mechanisms
        self._init_gates()
        
        # Initialize communication mechanism
        if enable_communication:
            self._init_communication(num_heads)
    
    def _init_gates(self):
        """Initialize the gating mechanisms based on gate_type."""
        gate_input_dim = self.state_dim * 2
        
        if self.gate_type == 'gru':
            # GRU-style gates
            self.reset_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.update_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.candidate_state = nn.Linear(gate_input_dim, self.state_dim)
        elif self.gate_type == 'lstm':
            # LSTM-style gates
            self.forget_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.input_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.output_gate = nn.Linear(gate_input_dim, self.state_dim)
            self.candidate_state = nn.Linear(gate_input_dim, self.state_dim)
        else:
            raise ValueError(f"Unsupported gate_type: {self.gate_type}. Use 'lstm' or 'gru'.")
    
    def _init_communication(self, num_heads: int):
        """
        Initialize the state-to-state communication mechanism.
        
        Args:
            num_heads: Number of attention heads
        """
        self.attention = nn.MultiheadAttention(
            embed_dim=self.state_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.state_dim)
    
    def forward(self, prev_state: torch.Tensor, new_input: torch.Tensor) -> torch.Tensor:
        """
        Apply gated update to propagate state with debugging.
        
        Args:
            prev_state: Previous state vector
            new_input: New input vector
            
        Returns:
            Updated state vector
        """
        if self.debug_mode:
            self.log_step("state_propagation_started", {
                'prev_state': prev_state,
                'new_input': new_input
            })
        
        # Handle both single state and multiple states
        if prev_state.dim() == 2:
            # Single state per batch
            updated_state = self._single_state_update(prev_state, new_input)
        else:
            # Multiple states per batch
            old_state = prev_state.clone()  # Store for logging
            updated_state = self._multi_state_update(prev_state, new_input)
            
            # Apply state-to-state communication if enabled
            if self.enable_communication:
                updated_state = self._apply_communication(updated_state)
            
            if self.debug_mode:
                self.log_state_update(old_state, updated_state, {
                    'gate_type': self.gate_type,
                    'communication_enabled': self.enable_communication
                })
        
        if self.debug_mode:
            self.log_step("state_propagation_completed", {
                'updated_state': updated_state
            })
        
        return updated_state
    
    def _single_state_update(self, prev_state: torch.Tensor, new_input: torch.Tensor) -> torch.Tensor:
        """
        Update a single state vector with debugging.
        
        Args:
            prev_state: Previous state [batch_size, state_dim]
            new_input: New input [batch_size, state_dim]
            
        Returns:
            Updated state [batch_size, state_dim]
        """
        if self.debug_mode:
            self.log_step("single_state_update", {
                'prev_state': prev_state,
                'new_input': new_input
            })
        
        if self.gate_type == 'gru':
            return self._gru_step(prev_state, new_input)
        else:  # lstm
            return self._lstm_step(prev_state, new_input)
    
    def _multi_state_update(self, prev_states: torch.Tensor, new_inputs: torch.Tensor) -> torch.Tensor:
        """
        Update multiple state vectors with debugging.
        
        Args:
            prev_states: Previous states [batch_size, num_states, state_dim]
            new_inputs: New inputs [batch_size, num_states, state_dim]
            
        Returns:
            Updated states [batch_size, num_states, state_dim]
        """
        if self.debug_mode:
            self.log_step("multi_state_update_started", {
                'prev_states': prev_states,
                'new_inputs': new_inputs
            })
        
        if self.gate_type == 'gru':
            updated_states = self._gru_step_multi(prev_states, new_inputs)
        else:  # lstm
            updated_states = self._lstm_step_multi(prev_states, new_inputs)
        
        if self.debug_mode:
            self.log_step("multi_state_update_completed", {
                'updated_states': updated_states
            })
        
        return updated_states
    
    def _gru_step(self, prev_state: torch.Tensor, new_input: torch.Tensor) -> torch.Tensor:
        """GRU-style gated update step with debugging."""
        if self.debug_mode:
            self.log_step("gru_step_started", {
                'prev_state': prev_state,
                'new_input': new_input
            })
        
        # Concatenate previous state and new input
        combined = torch.cat([prev_state, new_input], dim=1)
        
        # Calculate gates
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        update_gate = torch.sigmoid(self.update_gate(combined))
        
        # Calculate candidate state
        reset_prev_state = reset_gate * prev_state
        candidate_combined = torch.cat([reset_prev_state, new_input], dim=1)
        candidate_state = torch.tanh(self.candidate_state(candidate_combined))
        
        # Calculate new state using GRU formula
        new_state = (1 - update_gate) * prev_state + update_gate * candidate_state
        
        if self.debug_mode:
            self.log_step("gru_step_completed", {
                'reset_gate': reset_gate,
                'update_gate': update_gate,
                'candidate_state': candidate_state,
                'new_state': new_state
            })
        
        return new_state
    
    def _lstm_step(self, prev_state: torch.Tensor, new_input: torch.Tensor) -> torch.Tensor:
        """LSTM-style gated update step with debugging."""
        if self.debug_mode:
            self.log_step("lstm_step_started", {
                'prev_state': prev_state,
                'new_input': new_input
            })
        
        # Concatenate previous state and new input
        combined = torch.cat([prev_state, new_input], dim=1)
        
        # Calculate gates
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        
        # Calculate candidate values
        candidate_state = torch.tanh(self.candidate_state(combined))
        
        # Calculate new cell state
        cell_state = forget_gate * prev_state + input_gate * candidate_state
        
        # Calculate new hidden state
        new_state = output_gate * torch.tanh(cell_state)
        
        if self.debug_mode:
            self.log_step("lstm_step_completed", {
                'forget_gate': forget_gate,
                'input_gate': input_gate,
                'output_gate': output_gate,
                'candidate_state': candidate_state,
                'cell_state': cell_state,
                'new_state': new_state
            })
        
        return new_state
    
    def _gru_step_multi(self, prev_states: torch.Tensor, new_inputs: torch.Tensor) -> torch.Tensor:
        """GRU-style multi-state update with debugging."""
        if self.debug_mode:
            self.log_step("gru_multi_step_started", {
                'prev_states': prev_states,
                'new_inputs': new_inputs
            })
        
        batch_size, num_states, state_dim = prev_states.shape
        
        # Reshape for processing all states together
        prev_states_flat = prev_states.view(-1, state_dim)
        new_inputs_flat = new_inputs.view(-1, state_dim)
        
        # Concatenate previous states and new inputs
        combined = torch.cat([prev_states_flat, new_inputs_flat], dim=1)
        
        # Calculate gates
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        update_gate = torch.sigmoid(self.update_gate(combined))
        
        # Calculate candidate states
        reset_prev_states = reset_gate * prev_states_flat
        candidate_combined = torch.cat([reset_prev_states, new_inputs_flat], dim=1)
        candidate_states = torch.tanh(self.candidate_state(candidate_combined))
        
        # Calculate new states
        new_states = (1 - update_gate) * prev_states_flat + update_gate * candidate_states
        
        # Reshape back to original dimensions
        new_states = new_states.view(batch_size, num_states, state_dim)
        
        if self.debug_mode:
            self.log_step("gru_multi_step_completed", {
                'reset_gate': reset_gate,
                'update_gate': update_gate,
                'candidate_states': candidate_states,
                'new_states': new_states
            })
        
        return new_states
    
    def _lstm_step_multi(self, prev_states: torch.Tensor, new_inputs: torch.Tensor) -> torch.Tensor:
        """LSTM-style multi-state update with debugging."""
        if self.debug_mode:
            self.log_step("lstm_multi_step_started", {
                'prev_states': prev_states,
                'new_inputs': new_inputs
            })
        
        batch_size, num_states, state_dim = prev_states.shape
        
        # Reshape for processing all states together
        prev_states_flat = prev_states.view(-1, state_dim)
        new_inputs_flat = new_inputs.view(-1, state_dim)
        
        # Concatenate previous states and new inputs
        combined = torch.cat([prev_states_flat, new_inputs_flat], dim=1)
        
        # Calculate gates
        forget_gate = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        output_gate = torch.sigmoid(self.output_gate(combined))
        
        # Calculate candidate values
        candidate_states = torch.tanh(self.candidate_state(combined))
        
        # Calculate new cell states
        cell_states = forget_gate * prev_states_flat + input_gate * candidate_states
        
        # Calculate new hidden states
        new_states = output_gate * torch.tanh(cell_states)
        
        # Reshape back to original dimensions
        new_states = new_states.view(batch_size, num_states, state_dim)
        
        if self.debug_mode:
            self.log_step("lstm_multi_step_completed", {
                'forget_gate': forget_gate,
                'input_gate': input_gate,
                'output_gate': output_gate,
                'candidate_states': candidate_states,
                'cell_states': cell_states,
                'new_states': new_states
            })
        
        return new_states
    
    def _apply_communication(self, states: torch.Tensor) -> torch.Tensor:
        """
        Apply state-to-state communication with debugging.
        
        Args:
            states: State vectors [batch_size, num_states, state_dim]
            
        Returns:
            Updated state vectors [batch_size, num_states, state_dim]
        """
        if self.debug_mode:
            self.log_step("communication_started", {
                'states_before_communication': states
            })
        
        # Apply self-attention
        attended_states, attention_weights = self.attention(states, states, states)
        
        if self.debug_mode:
            self.log_step("attention_applied", {
                'attended_states': attended_states,
                'attention_weights': attention_weights
            })
            self.log_attention_operation(
                'state_to_state', f'states_{states.shape[1]}', f'states_{states.shape[1]}',
                attention_weights, attended_states
            )
        
        # Add residual connection and apply layer normalization
        updated_states = self.layer_norm(states + attended_states)
        
        if self.debug_mode:
            self.log_step("communication_completed", {
                'states_after_communication': updated_states
            })
        
        return updated_states


# Example usage
if __name__ == "__main__":
    print("Testing Debuggable NSM Components...")
    
    # Create debugger
    debugger = NSMDebugger("test_component_logs", verbose=True)
    debugger.enable_debug()
    
    # Test debuggable token-to-state router
    print("\n1. Testing DebuggableTokenToStateRouter...")
    router = DebuggableTokenToStateRouter(token_dim=64, state_dim=128, num_states=8, debug_mode=True)
    router.set_debugger(debugger)
    
    tokens = torch.randn(2, 10, 64)
    states = torch.randn(2, 8, 128)
    routed_tokens, routing_weights = router(tokens, states)
    print(f"   Input tokens: {tokens.shape} -> Routed tokens: {routed_tokens.shape}")
    
    # Test debuggable state manager
    print("\n2. Testing DebuggableStateManager...")
    state_manager = DebuggableStateManager(state_dim=128, max_states=16, initial_states=8, debug_mode=True)
    state_manager.set_debugger(debugger)
    
    active_states = state_manager()
    print(f"   Active states: {active_states.shape}")
    
    allocated = state_manager.allocate_states(2)
    print(f"   Allocated states: {allocated}")
    
    pruned = state_manager.prune_low_importance_states()
    print(f"   Pruned states: {pruned}")
    
    # Test debuggable state propagator
    print("\n3. Testing DebuggableStatePropagator...")
    propagator = DebuggableStatePropagator(state_dim=128, gate_type='gru', debug_mode=True)
    propagator.set_debugger(debugger)
    
    prev_state = torch.randn(2, 8, 128)
    new_input = torch.randn(2, 8, 128)
    updated_state = propagator(prev_state, new_input)
    print(f"   Previous state: {prev_state.shape} -> Updated state: {updated_state.shape}")
    
    # Print summary and save log
    debugger.print_summary()
    log_file = debugger.save_debug_log()
    print(f"\nDebug log saved to: {log_file}")
    
    print("\n✅ Debuggable NSM Components test completed!")