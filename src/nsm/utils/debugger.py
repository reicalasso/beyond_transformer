"""
Debug Mode for Neural State Machine

This module implements debug mode functionality for tracking state evolution
and monitoring memory operations in Neural State Machine models.
"""

import torch
import torch.nn as nn
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging


class NSMDebugger:
    """
    Debugger for Neural State Machine models.
    """
    
    def __init__(self, log_dir: str = "debug_logs", verbose: bool = True):
        """
        Initialize NSM debugger.
        
        Args:
            log_dir: Directory to save debug logs
            verbose: Whether to print debug information
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.verbose = verbose
        self.debug_enabled = False
        self.step_counter = 0
        self.log_data = []
        
        # Setup logging
        self.logger = logging.getLogger("NSMDebugger")
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Create file handler
        log_file = self.log_dir / f"nsm_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"NSM Debugger initialized. Logs saved to: {log_file}")
    
    def enable_debug(self):
        """Enable debug mode."""
        self.debug_enabled = True
        self.step_counter = 0
        self.log_data = []
        self.logger.info("Debug mode enabled")
    
    def disable_debug(self):
        """Disable debug mode."""
        self.debug_enabled = False
        self.logger.info("Debug mode disabled")
    
    def log_step(self, step_name: str, data: Dict[str, Any], 
                 step_info: Optional[Dict[str, Any]] = None):
        """
        Log a step in the processing pipeline.
        
        Args:
            step_name: Name of the step
            data: Data to log
            step_info: Additional step information
        """
        if not self.debug_enabled:
            return
        
        self.step_counter += 1
        
        # Create log entry
        log_entry = {
            'step': self.step_counter,
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'data': self._serialize_data(data),
            'step_info': step_info or {}
        }
        
        self.log_data.append(log_entry)
        
        if self.verbose:
            self.logger.debug(f"Step {self.step_counter}: {step_name}")
            self.logger.debug(f"  Data keys: {list(data.keys())}")
    
    def log_memory_operation(self, operation_type: str, 
                           memory_address: str,
                           read_data: Optional[torch.Tensor] = None,
                           write_data: Optional[torch.Tensor] = None,
                           attention_weights: Optional[torch.Tensor] = None,
                           operation_info: Optional[Dict[str, Any]] = None):
        """
        Log memory read/write operations.
        
        Args:
            operation_type: Type of operation ('read', 'write', 'erase', 'add')
            memory_address: Memory address or slot identifier
            read_data: Data read from memory
            write_data: Data written to memory
            attention_weights: Attention weights used for operation
            operation_info: Additional operation information
        """
        if not self.debug_enabled:
            return
        
        operation_data = {
            'operation_type': operation_type,
            'memory_address': memory_address,
            'read_data': self._serialize_tensor(read_data) if read_data is not None else None,
            'write_data': self._serialize_tensor(write_data) if write_data is not None else None,
            'attention_weights': self._serialize_tensor(attention_weights) if attention_weights is not None else None,
            'operation_info': operation_info or {}
        }
        
        self.log_step(f"Memory_{operation_type.capitalize()}", 
                     {'memory_operation': operation_data})
        
        if self.verbose:
            self.logger.debug(f"Memory {operation_type} at {memory_address}")
    
    def log_attention_operation(self, attention_type: str,
                              query_info: str,
                              key_info: str,
                              attention_weights: torch.Tensor,
                              attended_values: Optional[torch.Tensor] = None,
                              operation_info: Optional[Dict[str, Any]] = None):
        """
        Log attention operations.
        
        Args:
            attention_type: Type of attention operation
            query_info: Information about query
            key_info: Information about keys
            attention_weights: Attention weights tensor
            attended_values: Values after attention
            operation_info: Additional operation information
        """
        if not self.debug_enabled:
            return
        
        attention_data = {
            'attention_type': attention_type,
            'query_info': query_info,
            'key_info': key_info,
            'attention_weights': self._serialize_tensor(attention_weights),
            'attended_values': self._serialize_tensor(attended_values) if attended_values is not None else None,
            'operation_info': operation_info or {}
        }
        
        self.log_step(f"Attention_{attention_type.capitalize()}", 
                     {'attention_operation': attention_data})
        
        if self.verbose:
            self.logger.debug(f"Attention {attention_type}: {query_info} -> {key_info}")
    
    def log_state_update(self, component_name: str,
                        old_state: torch.Tensor,
                        new_state: torch.Tensor,
                        update_info: Optional[Dict[str, Any]] = None):
        """
        Log state updates.
        
        Args:
            component_name: Name of the component
            old_state: State before update
            new_state: State after update
            update_info: Additional update information
        """
        if not self.debug_enabled:
            return
        
        state_data = {
            'component_name': component_name,
            'old_state': self._serialize_tensor(old_state),
            'new_state': self._serialize_tensor(new_state),
            'state_diff': self._serialize_tensor(new_state - old_state) if old_state is not None else None,
            'update_info': update_info or {}
        }
        
        self.log_step(f"State_Update_{component_name}", 
                     {'state_update': state_data})
        
        # Log statistics
        if self.verbose and new_state is not None:
            new_state_np = new_state.detach().cpu().numpy()
            self.logger.debug(f"State update for {component_name}: "
                            f"mean={np.mean(new_state_np):.4f}, "
                            f"std={np.std(new_state_np):.4f}")
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize data for logging.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data
        """
        serialized = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                serialized[key] = self._serialize_tensor(value)
            elif isinstance(value, dict):
                serialized[key] = self._serialize_data(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = [self._serialize_data({'item': item})['item'] if isinstance(item, dict) 
                                 else self._serialize_tensor(item) if isinstance(item, torch.Tensor)
                                 else item for item in value]
            else:
                serialized[key] = value
        return serialized
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Serialize tensor for logging.
        
        Args:
            tensor: Tensor to serialize
            
        Returns:
            Dictionary with tensor information
        """
        if tensor is None:
            return None
            
        tensor_np = tensor.detach().cpu().numpy()
        return {
            'shape': list(tensor_np.shape),
            'dtype': str(tensor_np.dtype),
            'mean': float(np.mean(tensor_np)),
            'std': float(np.std(tensor_np)),
            'min': float(np.min(tensor_np)),
            'max': float(np.max(tensor_np)),
            'sample': tensor_np.flatten()[:100].tolist() if tensor_np.size > 0 else []  # First 100 elements
        }
    
    def save_debug_log(self, filename: Optional[str] = None) -> str:
        """
        Save debug log to file.
        
        Args:
            filename: Filename to save to (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"debug_log_{timestamp}.json"
        else:
            filename = Path(filename)
        
        # Add metadata
        log_with_metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_steps': self.step_counter,
            'log_data': self.log_data
        }
        
        with open(filename, 'w') as f:
            json.dump(log_with_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Debug log saved to: {filename}")
        return str(filename)
    
    def get_step_summary(self) -> Dict[str, int]:
        """
        Get summary of logged steps.
        
        Returns:
            Dictionary with step count by type
        """
        step_counts = {}
        for entry in self.log_data:
            step_name = entry['step_name']
            step_counts[step_name] = step_counts.get(step_name, 0) + 1
        return step_counts
    
    def print_summary(self):
        """Print debug summary."""
        if not self.debug_enabled:
            self.logger.info("Debug mode is not enabled")
            return
        
        step_summary = self.get_step_summary()
        
        self.logger.info("=== NSM Debug Summary ===")
        self.logger.info(f"Total steps logged: {self.step_counter}")
        self.logger.info("Step breakdown:")
        for step_name, count in step_summary.items():
            self.logger.info(f"  {step_name}: {count}")
        
        # Memory operations summary
        memory_ops = [entry for entry in self.log_data 
                     if 'memory_operation' in entry['data']]
        self.logger.info(f"Memory operations: {len(memory_ops)}")
        
        # Attention operations summary
        attention_ops = [entry for entry in self.log_data 
                       if 'attention_operation' in entry['data']]
        self.logger.info(f"Attention operations: {len(attention_ops)}")
        
        # State updates summary
        state_updates = [entry for entry in self.log_data 
                        if 'state_update' in entry['data']]
        self.logger.info(f"State updates: {len(state_updates)}")


class DebuggableNSMComponent(nn.Module):
    """
    Base class for debuggable NSM components.
    """
    
    def __init__(self, component_name: str, debug_mode: bool = False):
        """
        Initialize debuggable component.
        
        Args:
            component_name: Name of the component
            debug_mode: Whether to enable debug mode
        """
        super().__init__()
        self.component_name = component_name
        self.debug_mode = debug_mode
        self.debugger = None
    
    def set_debugger(self, debugger: NSMDebugger):
        """
        Set debugger for this component.
        
        Args:
            debugger: NSMDebugger instance
        """
        self.debugger = debugger
        self.debug_mode = True
    
    def log_step(self, step_name: str, data: Dict[str, Any], 
                 step_info: Optional[Dict[str, Any]] = None):
        """
        Log a step if debugger is available.
        
        Args:
            step_name: Name of the step
            data: Data to log
            step_info: Additional step information
        """
        if self.debugger is not None:
            self.debugger.log_step(f"{self.component_name}_{step_name}", data, step_info)
    
    def log_memory_operation(self, operation_type: str, 
                           memory_address: str,
                           read_data: Optional[torch.Tensor] = None,
                           write_data: Optional[torch.Tensor] = None,
                           attention_weights: Optional[torch.Tensor] = None,
                           operation_info: Optional[Dict[str, Any]] = None):
        """
        Log memory operation if debugger is available.
        
        Args:
            operation_type: Type of operation
            memory_address: Memory address
            read_data: Data read from memory
            write_data: Data written to memory
            attention_weights: Attention weights
            operation_info: Additional operation information
        """
        if self.debugger is not None:
            self.debugger.log_memory_operation(
                operation_type, memory_address, read_data, write_data,
                attention_weights, operation_info
            )
    
    def log_attention_operation(self, attention_type: str,
                              query_info: str,
                              key_info: str,
                              attention_weights: torch.Tensor,
                              attended_values: Optional[torch.Tensor] = None,
                              operation_info: Optional[Dict[str, Any]] = None):
        """
        Log attention operation if debugger is available.
        
        Args:
            attention_type: Type of attention operation
            query_info: Information about query
            key_info: Information about keys
            attention_weights: Attention weights tensor
            attended_values: Values after attention
            operation_info: Additional operation information
        """
        if self.debugger is not None:
            self.debugger.log_attention_operation(
                attention_type, query_info, key_info, attention_weights,
                attended_values, operation_info
            )
    
    def log_state_update(self, old_state: torch.Tensor,
                        new_state: torch.Tensor,
                        update_info: Optional[Dict[str, Any]] = None):
        """
        Log state update if debugger is available.
        
        Args:
            old_state: State before update
            new_state: State after update
            update_info: Additional update information
        """
        if self.debugger is not None:
            self.debugger.log_state_update(
                self.component_name, old_state, new_state, update_info
            )


# Example usage
if __name__ == "__main__":
    print("Testing NSM Debugger...")
    
    # Create debugger
    debugger = NSMDebugger("test_debug_logs", verbose=True)
    debugger.enable_debug()
    
    # Test logging
    sample_tensor = torch.randn(4, 8)
    debugger.log_step("test_step", {
        'input_data': sample_tensor,
        'processing_info': {'layer': 'test', 'operation': 'forward'}
    })
    
    # Test memory operation logging
    debugger.log_memory_operation(
        'read', 'memory_slot_0',
        read_data=torch.randn(8),
        attention_weights=torch.softmax(torch.randn(8), dim=0)
    )
    
    # Test attention operation logging
    debugger.log_attention_operation(
        'token_to_state', 'token_5', 'state_2',
        attention_weights=torch.softmax(torch.randn(4), dim=0),
        attended_values=torch.randn(4)
    )
    
    # Test state update logging
    old_state = torch.randn(6, 12)
    new_state = old_state + torch.randn(6, 12) * 0.1
    debugger.log_state_update('test_component', old_state, new_state)
    
    # Print summary
    debugger.print_summary()
    
    # Save log
    log_file = debugger.save_debug_log()
    print(f"Debug log saved to: {log_file}")
    
    print("✅ NSM Debugger test completed!")