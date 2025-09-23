"""
Test script for debug mode components
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Test imports
try:
    # Try absolute imports first
    from nsm.modules.debuggable_components import (
        DebuggableStateManager,
        DebuggableStatePropagator,
        DebuggableTokenToStateRouter,
    )
    from nsm.utils.debugger import NSMDebugger
except ImportError:
    # Fall back to relative imports
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from modules.debuggable_components import (
        DebuggableStateManager,
        DebuggableStatePropagator,
        DebuggableTokenToStateRouter,
    )
    from utils.debugger import NSMDebugger

    print("✅ All debug mode imports successful")

    # Test debugger
    debugger = NSMDebugger("test_debug_logs", verbose=False)
    debugger.enable_debug()
    print("✅ Debugger initialized and enabled")

    # Test logging
    sample_tensor = torch.randn(4, 8)
    debugger.log_step(
        "test_step",
        {
            "input_data": sample_tensor,
            "processing_info": {"layer": "test", "operation": "forward"},
        },
    )
    print("✅ Step logging working")

    # Test memory operation logging
    debugger.log_memory_operation(
        "read",
        "memory_slot_0",
        read_data=torch.randn(8),
        attention_weights=torch.softmax(torch.randn(8), dim=0),
    )
    print("✅ Memory operation logging working")

    # Test attention operation logging
    debugger.log_attention_operation(
        "token_to_state",
        "token_5",
        "state_2",
        attention_weights=torch.softmax(torch.randn(4), dim=0),
        attended_values=torch.randn(4),
    )
    print("✅ Attention operation logging working")

    # Test state update logging
    old_state = torch.randn(6, 12)
    new_state = old_state + torch.randn(6, 12) * 0.1
    debugger.log_state_update("test_component", old_state, new_state)
    print("✅ State update logging working")

    # Test debuggable components
    router = DebuggableTokenToStateRouter(
        token_dim=32, state_dim=64, num_states=4, debug_mode=True
    )
    router.set_debugger(debugger)
    print("✅ Debuggable router created")

    state_manager = DebuggableStateManager(
        state_dim=64, max_states=8, initial_states=4, debug_mode=True
    )
    state_manager.set_debugger(debugger)
    print("✅ Debuggable state manager created")

    propagator = DebuggableStatePropagator(
        state_dim=64, gate_type="gru", debug_mode=True
    )
    propagator.set_debugger(debugger)
    print("✅ Debuggable propagator created")

    # Test forward passes
    tokens = torch.randn(2, 5, 32)
    states = torch.randn(2, 4, 64)
    routed_tokens, routing_weights = router(tokens, states)
    print("✅ Router forward pass successful")

    active_states = state_manager()
    print("✅ State manager forward pass successful")

    updated_states = propagator(states, routed_tokens)
    print("✅ Propagator forward pass successful")

    # Save log
    log_file = debugger.save_debug_log()
    print(f"✅ Debug log saved to: {log_file}")

    print("\n🎉 All debug mode components working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
