"""
Unit tests for visualization tools.
"""

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from nsm.utils.advanced_visualizer import AdvancedNSMVisualizer
from nsm.utils.visualizer import NSMVisualizer


class TestNSMVisualizer:
    """Test suite for NSMVisualizer."""

    @pytest.fixture
    def visualizer(self):
        """Create a NSMVisualizer for testing."""
        return NSMVisualizer(figsize=(8, 6), style="whitegrid")

    def test_initialization(self):
        """Test NSMVisualizer initialization."""
        visualizer = NSMVisualizer(figsize=(10, 8), style="darkgrid")

        assert visualizer.figsize == (10, 8)
        assert hasattr(visualizer, "palette")

    def test_plot_attention_map(self, visualizer):
        """Test plotting attention map."""
        attention_weights = torch.softmax(torch.randn(8, 8), dim=-1)

        # Test without saving
        with patch("matplotlib.pyplot.show"):
            fig = visualizer.plot_attention_map(
                attention_weights,
                title="Test Attention Map",
                x_labels=[f"X{i}" for i in range(8)],
                y_labels=[f"Y{i}" for i in range(8)],
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_memory_content(self, visualizer):
        """Test plotting memory content."""
        memory_content = torch.randn(16, 20)

        with patch("matplotlib.pyplot.show"):
            fig = visualizer.plot_memory_content(
                memory_content, title="Test Memory Content"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_state_evolution(self, visualizer):
        """Test plotting state evolution."""
        states = [torch.randn(8, 16) for _ in range(5)]

        with patch("matplotlib.pyplot.show"):
            fig = visualizer.plot_state_evolution(states, title="Test State Evolution")

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_memory_importance(self, visualizer):
        """Test plotting memory importance."""
        importance_scores = torch.sigmoid(torch.randn(16))

        with patch("matplotlib.pyplot.show"):
            fig = visualizer.plot_memory_importance(
                importance_scores, title="Test Memory Importance"
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_create_interactive_summary(self, visualizer):
        """Test creating interactive summary."""
        test_data = {
            "attention_weights": torch.softmax(torch.randn(8, 8), dim=-1),
            "memory_content": torch.randn(16, 20),
            "importance_scores": torch.sigmoid(torch.randn(16)),
        }

        summary_df = visualizer.create_interactive_summary(test_data)

        assert summary_df is not None
        assert len(summary_df) >= 1
        assert "Metric" in summary_df.columns
        assert "Mean" in summary_df.columns


class TestAdvancedNSMVisualizer:
    """Test suite for AdvancedNSMVisualizer."""

    @pytest.fixture
    def advanced_visualizer(self):
        """Create an AdvancedNSMVisualizer for testing."""
        return AdvancedNSMVisualizer(figsize=(10, 8), style="whitegrid")

    def test_initialization(self):
        """Test AdvancedNSMVisualizer initialization."""
        visualizer = AdvancedNSMVisualizer(figsize=(12, 10), style="darkgrid")

        assert visualizer.figsize == (12, 10)
        assert hasattr(visualizer, "palette")

    def test_plot_token_to_state_routing(self, advanced_visualizer):
        """Test plotting token-to-state routing."""
        routing_weights = torch.softmax(torch.randn(12, 8), dim=-1)

        with patch("matplotlib.pyplot.show"):
            fig = advanced_visualizer.plot_token_to_state_routing(
                routing_weights,
                token_labels=[f"T{i}" for i in range(12)],
                state_labels=[f"S{i}" for i in range(8)],
                title="Test Token-to-State Routing",
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_state_communication(self, advanced_visualizer):
        """Test plotting state communication."""
        state_attention = torch.softmax(torch.randn(8, 8), dim=-1)

        with patch("matplotlib.pyplot.show"):
            fig = advanced_visualizer.plot_state_communication(
                state_attention,
                state_labels=[f"State{i}" for i in range(8)],
                title="Test State Communication",
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_memory_read_write_operations(self, advanced_visualizer):
        """Test plotting memory read/write operations."""
        read_weights = torch.softmax(torch.randn(10), dim=0)
        write_weights = torch.softmax(torch.randn(10), dim=0)

        with patch("matplotlib.pyplot.show"):
            fig = advanced_visualizer.plot_memory_read_write_operations(
                read_weights,
                write_weights,
                memory_slots=[f"Slot{i}" for i in range(10)],
                title="Test Memory Operations",
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_plot_state_dynamics(self, advanced_visualizer):
        """Test plotting state dynamics."""
        state_trajectories = [torch.randn(6, 12) for _ in range(8)]

        with patch("matplotlib.pyplot.show"):
            fig = advanced_visualizer.plot_state_dynamics(
                state_trajectories,
                state_labels=[f"S{i}" for i in range(6)],
                metrics=["norm", "mean"],
                title="Test State Dynamics",
            )

        assert fig is not None
        assert isinstance(fig, plt.Figure)

    def test_create_summary_statistics(self, advanced_visualizer):
        """Test creating summary statistics."""
        test_data = {
            "attention_weights": torch.softmax(torch.randn(8, 8), dim=-1),
            "routing_weights": torch.softmax(torch.randn(12, 8), dim=-1),
            "memory_content": torch.randn(16, 20),
        }

        summary_df = advanced_visualizer.create_summary_statistics(test_data)

        assert summary_df is not None
        assert len(summary_df) >= 1
        assert "Component" in summary_df.columns
        assert "Mean" in summary_df.columns

    @pytest.mark.slow
    def test_create_comprehensive_report(self, advanced_visualizer, temp_dir):
        """Test creating comprehensive report."""
        test_data = {
            "attention_weights": torch.softmax(torch.randn(8, 8), dim=-1),
            "routing_weights": torch.softmax(torch.randn(12, 8), dim=-1),
            "memory_content": torch.randn(16, 20),
            "importance_scores": torch.sigmoid(torch.randn(16)),
        }

        report_dir = advanced_visualizer.create_comprehensive_report(
            test_data, save_dir=str(temp_dir)
        )

        assert report_dir is not None
        assert isinstance(report_dir, str)


# Integration tests for visualization
class TestVisualizationIntegration:
    """Integration tests for visualization tools."""

    @pytest.mark.integration
    def test_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create visualizer
        visualizer = NSMVisualizer()
        advanced_visualizer = AdvancedNSMVisualizer()

        # Generate test data
        attention_weights = torch.softmax(torch.randn(8, 8), dim=-1)
        memory_content = torch.randn(16, 20)
        routing_weights = torch.softmax(torch.randn(12, 8), dim=-1)
        state_attention = torch.softmax(torch.randn(8, 8), dim=-1)

        test_data = {
            "attention_weights": attention_weights,
            "memory_content": memory_content,
            "routing_weights": routing_weights,
            "state_attention": state_attention,
        }

        # Test basic visualizations
        with patch("matplotlib.pyplot.show"):
            fig1 = visualizer.plot_attention_map(attention_weights)
            fig2 = visualizer.plot_memory_content(memory_content)
            fig3 = advanced_visualizer.plot_token_to_state_routing(routing_weights)
            fig4 = advanced_visualizer.plot_state_communication(state_attention)

        # Test summary creation
        summary_df = visualizer.create_interactive_summary(test_data)
        advanced_summary_df = advanced_visualizer.create_summary_statistics(test_data)

        # All should succeed
        assert all(fig is not None for fig in [fig1, fig2, fig3, fig4])
        assert summary_df is not None
        assert advanced_summary_df is not None

    @pytest.mark.slow
    def test_large_data_visualization(self):
        """Test visualization with larger datasets."""
        visualizer = NSMVisualizer()

        # Large attention matrix
        large_attention = torch.softmax(torch.randn(32, 32), dim=-1)
        large_memory = torch.randn(64, 128)

        with patch("matplotlib.pyplot.show"):
            # Should handle large data without crashing
            fig1 = visualizer.plot_attention_map(large_attention)
            fig2 = visualizer.plot_memory_content(large_memory)

        assert fig1 is not None
        assert fig2 is not None

    def test_error_handling(self):
        """Test visualization error handling."""
        visualizer = NSMVisualizer()

        # Test with invalid data
        with patch("matplotlib.pyplot.show"):
            # Empty tensor
            empty_tensor = torch.tensor([])

            # Should handle gracefully
            try:
                fig = visualizer.plot_attention_map(empty_tensor)
                # Might return None or handle gracefully
            except Exception as e:
                # Should be handled exceptions, not crashes
                assert isinstance(e, (ValueError, RuntimeError))


# Performance tests for visualization
class TestVisualizationPerformance:
    """Performance tests for visualization tools."""

    @pytest.mark.slow
    def test_plotting_speed(self):
        """Test plotting speed performance."""
        visualizer = NSMVisualizer()

        # Medium-sized data
        attention_weights = torch.softmax(torch.randn(16, 16), dim=-1)
        memory_content = torch.randn(32, 64)

        import time

        # Time attention plot
        start_time = time.time()
        with patch("matplotlib.pyplot.show"):
            _ = visualizer.plot_attention_map(attention_weights)
        attention_time = time.time() - start_time

        # Time memory plot
        start_time = time.time()
        with patch("matplotlib.pyplot.show"):
            _ = visualizer.plot_memory_content(memory_content)
        memory_time = time.time() - start_time

        # Should be reasonably fast (less than 2 seconds each)
        assert (
            attention_time < 2.0
        ), f"Attention plotting too slow: {attention_time:.4f}s"
        assert memory_time < 2.0, f"Memory plotting too slow: {memory_time:.4f}s"

    def test_memory_efficiency(self):
        """Test memory efficiency of visualization."""
        import gc

        visualizer = NSMVisualizer()

        # Small data first
        small_attention = torch.softmax(torch.randn(8, 8), dim=-1)

        # Clear memory
        gc.collect()

        with patch("matplotlib.pyplot.show"):
            # Create plot
            fig = visualizer.plot_attention_map(small_attention)

        # Clear plot
        plt.close(fig)
        gc.collect()

        # Should not cause memory issues
        assert True  # Test passes if no memory errors


# Test utility functions
class TestVisualizationUtilities:
    """Test utility functions for visualization."""

    def test_color_palette_consistency(self):
        """Test that color palettes are consistent."""
        viz1 = NSMVisualizer()
        viz2 = NSMVisualizer()

        # Palettes should be the same
        assert len(viz1.palette) == len(viz2.palette)

        # Colors should be consistent
        for c1, c2 in zip(viz1.palette, viz2.palette):
            assert len(c1) == len(c2)

    def test_figure_saving(self, temp_dir):
        """Test figure saving functionality."""
        visualizer = NSMVisualizer()
        attention_weights = torch.softmax(torch.randn(8, 8), dim=-1)

        save_path = temp_dir / "test_plot.png"

        with patch("matplotlib.pyplot.show"):
            fig = visualizer.plot_attention_map(
                attention_weights, save_path=str(save_path)
            )

        # Check that file was created
        assert save_path.exists()

        # Check that file is not empty
        assert save_path.stat().st_size > 0

    def test_data_serialization(self, temp_dir):
        """Test data serialization for reports."""
        visualizer = NSMVisualizer()

        test_data = {
            "attention_weights": torch.softmax(torch.randn(8, 8), dim=-1),
            "memory_content": torch.randn(16, 20),
        }

        save_path = temp_dir / "test_summary.csv"
        summary_df = visualizer.create_interactive_summary(
            test_data, save_path=str(save_path)
        )

        # Check that CSV was created
        assert save_path.exists()

        # Check that CSV is not empty
        assert save_path.stat().st_size > 0

        # Check that dataframe is valid
        assert len(summary_df) >= 1
        assert "Metric" in summary_df.columns
