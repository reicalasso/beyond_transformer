# API Reference

This document provides detailed documentation for all public APIs in the Neural State Machine (NSM) library.

## Core Modules

### nsm.StatePropagator

```python
class StatePropagator(nn.Module):
    def __init__(self, state_dim: int, gate_type: str = 'gru', 
                 num_heads: int = 4, enable_communication: bool = True):
        """
        Initialize the StatePropagator.
        
        Args:
            state_dim (int): Dimension of the state vectors
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
            num_heads (int): Number of attention heads for state-to-state communication
            enable_communication (bool): Whether to enable state-to-state communication
        """
        pass

    def forward(self, prev_state: torch.Tensor, new_input: torch.Tensor) -> torch.Tensor:
        """
        Apply gated update to propagate state.
        
        Args:
            prev_state (torch.Tensor): Previous state vector(s)
                Shape: [batch_size, state_dim] or [batch_size, num_states, state_dim]
            new_input (torch.Tensor): New input vector(s)
                Shape: [batch_size, state_dim] or [batch_size, num_states, state_dim]
                
        Returns:
            torch.Tensor: Updated state vector(s) with same shape as input
        """
        pass
```

#### Parameters

- **state_dim** (`int`): Dimension of the state vectors. This determines the size of each state embedding.
- **gate_type** (`str`, optional): Type of gating mechanism to use. Options are `'gru'` (default) or `'lstm'`.
- **num_heads** (`int`, optional): Number of attention heads for state-to-state communication. Default is 4.
- **enable_communication** (`bool`, optional): Whether to enable state-to-state communication using MultiHeadAttention. Default is True.

#### Methods

##### `forward(prev_state, new_input)`
Applies gated update to propagate state.

**Parameters:**
- **prev_state** (`torch.Tensor`): Previous state vector(s)
  - Single state: `[batch_size, state_dim]`
  - Multiple states: `[batch_size, num_states, state_dim]`
- **new_input** (`torch.Tensor`): New input vector(s) with matching dimensions to `prev_state`

**Returns:**
- `torch.Tensor`: Updated state vector(s) with same shape as input

#### Example Usage

```python
import torch
from nsm import StatePropagator

# Single state propagation
propagator = StatePropagator(state_dim=128, gate_type='gru')
prev_state = torch.randn(32, 128)  # batch_size=32
new_input = torch.randn(32, 128)
updated_state = propagator(prev_state, new_input)

# Multiple state propagation with communication
propagator_multi = StatePropagator(state_dim=128, gate_type='lstm', num_heads=8)
prev_states = torch.randn(32, 16, 128)  # 16 states per batch
new_inputs = torch.randn(32, 16, 128)
updated_states = propagator_multi(prev_states, new_inputs)
```

### nsm.TokenToStateRouter

```python
class TokenToStateRouter(nn.Module):
    def __init__(self, token_dim: int, state_dim: int, num_states: int, num_heads: int = 4):
        """
        Initialize the TokenToStateRouter.
        
        Args:
            token_dim (int): Dimension of input tokens
            state_dim (int): Dimension of state vectors
            num_states (int): Number of state nodes
            num_heads (int): Number of routing heads
        """
        pass

    def forward(self, tokens: torch.Tensor, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route tokens to states.
        
        Args:
            tokens (torch.Tensor): Input tokens [batch_size, seq_len, token_dim]
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Routed tokens [batch_size, num_states, state_dim]
                - Routing weights [batch_size, seq_len, num_states]
        """
        pass
```

#### Parameters

- **token_dim** (`int`): Dimension of input tokens
- **state_dim** (`int`): Dimension of state vectors
- **num_states** (`int`): Number of state nodes
- **num_heads** (`int`, optional): Number of routing heads. Default is 4.

#### Methods

##### `forward(tokens, states)`
Route tokens to states.

**Parameters:**
- **tokens** (`torch.Tensor`): Input tokens with shape `[batch_size, seq_len, token_dim]`
- **states** (`torch.Tensor`): State vectors with shape `[batch_size, num_states, state_dim]`

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: 
  - **Routed tokens**: `[batch_size, num_states, state_dim]`
  - **Routing weights**: `[batch_size, seq_len, num_states]`

### nsm.StateManager

```python
class StateManager(nn.Module):
    def __init__(self, state_dim: int, max_states: int = 64, initial_states: Optional[int] = None, 
                 prune_threshold: float = 0.1):
        """
        Initialize the StateManager.
        
        Args:
            state_dim (int): Dimension of each state vector
            max_states (int): Maximum number of state nodes
            initial_states (int, optional): Initial number of active states
            prune_threshold (float): Threshold for pruning states (0-1)
        """
        pass

    def forward(self) -> torch.Tensor:
        """
        Get current active states.
        
        Returns:
            torch.Tensor: Active state vectors [num_active_states, state_dim]
        """
        pass

    def get_importance_scores(self) -> torch.Tensor:
        """
        Get importance scores for all states.
        
        Returns:
            torch.Tensor: Importance scores [max_states]
        """
        pass

    def get_active_count(self) -> int:
        """
        Get number of currently active states.
        
        Returns:
            int: Number of active states
        """
        pass

    def prune_low_importance_states(self) -> int:
        """
        Prune states below importance threshold.
        
        Returns:
            int: Number of states pruned
        """
        pass

    def allocate_states(self, num_states: int) -> int:
        """
        Allocate additional state nodes.
        
        Args:
            num_states (int): Number of states to allocate
            
        Returns:
            int: Number of states actually allocated
        """
        pass
```

#### Parameters

- **state_dim** (`int`): Dimension of each state vector
- **max_states** (`int`, optional): Maximum number of state nodes. Default is 64.
- **initial_states** (`int`, optional): Initial number of active states. If None, uses `max_states`.
- **prune_threshold** (`float`, optional): Threshold for pruning states (0-1). Default is 0.1.

#### Methods

##### `forward()`
Get current active states.

**Returns:**
- `torch.Tensor`: Active state vectors with shape `[num_active_states, state_dim]`

##### `get_importance_scores()`
Get importance scores for all states.

**Returns:**
- `torch.Tensor`: Importance scores with shape `[max_states]`

##### `get_active_count()`
Get number of currently active states.

**Returns:**
- `int`: Number of active states

##### `prune_low_importance_states()`
Prune states below importance threshold.

**Returns:**
- `int`: Number of states pruned

##### `allocate_states(num_states)`
Allocate additional state nodes.

**Parameters:**
- **num_states** (`int`): Number of states to allocate

**Returns:**
- `int`: Number of states actually allocated

### nsm.NSMLayer

```python
class NSMLayer(nn.Module):
    def __init__(self, state_dim: int, token_dim: int, num_heads: int = 4):
        """
        Initialize the NSMLayer.
        
        Args:
            state_dim (int): Dimension of state vectors
            token_dim (int): Dimension of input tokens
            num_heads (int): Number of attention heads
        """
        pass

    def forward(self, states: torch.Tensor, tokens: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the NSMLayer.
        
        Args:
            states (torch.Tensor): State vectors [batch_size, num_states, state_dim]
            tokens (torch.Tensor): Input tokens [batch_size, seq_len, token_dim]
            attention_mask (torch.Tensor, optional): Attention mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Updated states [batch_size, num_states, state_dim]
        """
        pass
```

#### Parameters

- **state_dim** (`int`): Dimension of state vectors
- **token_dim** (`int`): Dimension of input tokens
- **num_heads** (`int`): Number of attention heads

#### Methods

##### `forward(states, tokens, attention_mask=None)`
Forward pass of the NSMLayer.

**Parameters:**
- **states** (`torch.Tensor`): State vectors with shape `[batch_size, num_states, state_dim]`
- **tokens** (`torch.Tensor`): Input tokens with shape `[batch_size, seq_len, token_dim]`
- **attention_mask** (`torch.Tensor`, optional): Attention mask with shape `[batch_size, seq_len]`

**Returns:**
- `torch.Tensor`: Updated states with shape `[batch_size, num_states, state_dim]`

## Models

### nsm.models.SimpleNSM

```python
class SimpleNSM(nn.Module):
    def __init__(self, input_dim: int, state_dim: int, num_states: int, output_dim: int, gate_type: str = 'gru'):
        """
        Initialize the SimpleNSM model.
        
        Args:
            input_dim (int): Dimension of input features
            state_dim (int): Dimension of state vectors
            num_states (int): Number of state nodes
            output_dim (int): Dimension of output
            gate_type (str): Type of gating mechanism ('lstm' or 'gru')
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        pass
```

#### Parameters

- **input_dim** (`int`): Dimension of input features
- **state_dim** (`int`): Dimension of state vectors
- **num_states** (`int`): Number of state nodes
- **output_dim** (`int`): Dimension of output
- **gate_type** (`str`, optional): Type of gating mechanism ('lstm' or 'gru'). Default is 'gru'.

#### Methods

##### `forward(x)`
Forward pass of the model.

**Parameters:**
- **x** (`torch.Tensor`): Input tensor with shape `[batch_size, input_dim]`

**Returns:**
- `torch.Tensor`: Output tensor with shape `[batch_size, output_dim]`

## Utilities

### nsm.utils.debugger.NSMDebugger

```python
class NSMDebugger:
    def __init__(self, log_dir: str = "debug_logs", verbose: bool = True):
        """
        Initialize NSM debugger.
        
        Args:
            log_dir (str): Directory to save debug logs
            verbose (bool): Whether to print debug information
        """
        pass

    def enable_debug(self):
        """Enable debug mode."""
        pass

    def disable_debug(self):
        """Disable debug mode."""
        pass

    def log_step(self, step_name: str, data: Dict[str, Any], 
                 step_info: Optional[Dict[str, Any]] = None):
        """
        Log a step in the processing pipeline.
        
        Args:
            step_name (str): Name of the step
            data (Dict[str, Any]): Data to log
            step_info (Dict[str, Any], optional): Additional step information
        """
        pass

    def log_memory_operation(self, operation_type: str, 
                           memory_address: str,
                           read_data: Optional[torch.Tensor] = None,
                           write_data: Optional[torch.Tensor] = None,
                           attention_weights: Optional[torch.Tensor] = None,
                           operation_info: Optional[Dict[str, Any]] = None):
        """
        Log memory read/write operations.
        
        Args:
            operation_type (str): Type of operation ('read', 'write', 'erase', 'add')
            memory_address (str): Memory address or slot identifier
            read_data (torch.Tensor, optional): Data read from memory
            write_data (torch.Tensor, optional): Data written to memory
            attention_weights (torch.Tensor, optional): Attention weights used for operation
            operation_info (Dict[str, Any], optional): Additional operation information
        """
        pass

    def save_debug_log(self, filename: Optional[str] = None) -> str:
        """
        Save debug log to file.
        
        Args:
            filename (str, optional): Filename to save to
            
        Returns:
            str: Path to saved file
        """
        pass
```

This API reference provides comprehensive documentation for all public classes and methods in the NSM library. Each component includes detailed parameter descriptions, return value information, and usage examples.

For more detailed information about specific components, please refer to their respective module documentation.