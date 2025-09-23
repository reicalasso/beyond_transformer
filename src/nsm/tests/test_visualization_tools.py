"""
Test script for visualization tools
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Test imports
try:
    # Try absolute imports first
    from nsm.utils.advanced_visualizer import AdvancedNSMVisualizer
    from nsm.utils.visualizer import NSMVisualizer
except ImportError:
    # Fall back to relative imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from utils.advanced_visualizer import AdvancedNSMVisualizer
    from utils.visualizer import NSMVisualizer

    print("✅ All visualization imports successful")

    # Test basic visualizer
    visualizer = NSMVisualizer()

    # Create sample data
    attention_weights = torch.softmax(torch.randn(6, 6), dim=-1)
    memory_content = torch.randn(8, 12)

    # Test basic plots (without displaying)
    fig1 = visualizer.plot_attention_map(attention_weights, title="Test Attention")
    print("✅ Attention map visualization working")

    fig2 = visualizer.plot_memory_content(memory_content, title="Test Memory")
    print("✅ Memory content visualization working")

    # Test advanced visualizer
    advanced_visualizer = AdvancedNSMVisualizer()

    routing_weights = torch.softmax(torch.randn(10, 6), dim=-1)
    state_attention = torch.softmax(torch.randn(6, 6), dim=-1)

    fig3 = advanced_visualizer.plot_token_to_state_routing(routing_weights)
    print("✅ Token-to-state routing visualization working")

    fig4 = advanced_visualizer.plot_state_communication(state_attention)
    print("✅ State communication visualization working")

    # Test data summary
    test_data = {
        "attention_weights": attention_weights,
        "memory_content": memory_content,
    }
    summary_df = visualizer.create_interactive_summary(test_data)
    print("✅ Interactive summary creation working")

    print("\n🎉 All visualization tools working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
