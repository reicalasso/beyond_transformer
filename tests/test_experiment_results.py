"""
Tests for experiment results management utilities.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pulse.experiments.experiment_results import ExperimentResults


def test_experiment_results_basic():
    """Test basic ExperimentResults functionality."""
    # Create a temporary directory for test results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create experiment results manager
        results_manager = ExperimentResults(temp_dir)

        # Test directory creation
        assert (Path(temp_dir) / "experiments").exists()
        assert (Path(temp_dir) / "processed").exists()
        assert (Path(temp_dir) / "visualizations").exists()
        assert (Path(temp_dir) / "logs").exists()
        assert (Path(temp_dir) / "summaries").exists()

        # Test saving results
        example_results = {
            "state_counts": [8, 16, 32, 64],
            "accuracies": [53.2, 65.8, 72.1, 75.3],
            "memory_usages": [100.5, 200.1, 400.3, 800.7],
            "training_times": [120.5, 240.1, 480.3, 960.7],
        }

        filepath = results_manager.save_experiment_results(
            "test_experiment", example_results
        )
        assert os.path.exists(filepath)

        # Test loading results
        loaded_results = results_manager.load_experiment_results("test_experiment")
        assert loaded_results["experiment_name"] == "test_experiment"
        assert (
            loaded_results["results"]["state_counts"] == example_results["state_counts"]
        )

        # Test listing experiments
        experiments = results_manager.list_experiments()
        assert "test_experiment" in experiments

        # Test getting DataFrame
        df = results_manager.get_experiment_results_df("test_experiment")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 4 state counts
        assert list(df.columns) == [
            "state_counts",
            "accuracies",
            "memory_usages",
            "training_times",
        ]

        print("✓ ExperimentResults basic test passed")


def test_experiment_results_processed():
    """Test saving processed results."""
    # Create a temporary directory for test results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create experiment results manager
        results_manager = ExperimentResults(temp_dir)

        # Test saving processed results as dict
        processed_data = {"mean_accuracy": 66.6, "std_accuracy": 9.8}
        filepath = results_manager.save_processed_results(
            "test_processed", processed_data
        )
        assert os.path.exists(filepath)

        # Test saving processed results as DataFrame
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        filepath = results_manager.save_processed_results("test_df", df)
        assert os.path.exists(filepath)

        print("✓ ExperimentResults processed test passed")


def test_experiment_results_other():
    """Test saving logs, visualizations, and summaries."""
    # Create a temporary directory for test results
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create experiment results manager
        results_manager = ExperimentResults(temp_dir)

        # Test saving log
        log_content = "This is a test log entry."
        filepath = results_manager.save_log("test_log", log_content)
        assert os.path.exists(filepath)

        # Verify content
        with open(filepath, "r") as f:
            assert f.read() == log_content

        # Test saving summary
        summary_content = "# Test Summary\n\nThis is a test summary."
        filepath = results_manager.save_summary("test_summary", summary_content)
        assert os.path.exists(filepath)

        # Verify content
        with open(filepath, "r") as f:
            assert f.read() == summary_content

        print("✓ ExperimentResults other test passed")


def run_all_tests():
    """Run all experiment results tests."""
    print("Running Experiment Results Tests...")
    print("=" * 35)

    test_experiment_results_basic()
    test_experiment_results_processed()
    test_experiment_results_other()

    print("\n✓ All experiment results tests passed!")


if __name__ == "__main__":
    run_all_tests()
