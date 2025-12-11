"""
Small-Scale Tests for PULSE Models

This module implements small-scale tests on synthetic data with logging
of memory content, attention weights, and state variables.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pulse.data.synthetic_data import SyntheticDataGenerator
from pulse.models import AdvancedHybridModel, SequentialHybridModel, SimplePulse
from pulse.utils.logger import PULSELogger


class SmallScaleTester:
    """
    Small-scale tester for PULSE models.
    """

    def __init__(self, log_dir: str = "test_logs"):
        """
        Initialize the SmallScaleTester.

        Args:
            log_dir: Directory to save test logs
        """
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def test_simple_pulse(self):
        """
        Test SimplePulse model on synthetic data.
        """
        print("\n" + "=" * 50)
        print("Testing SimplePulse Model")
        print("=" * 50)

        # Create logger
        logger = PULSELogger(self.log_dir, "simple_pulse_test")

        # Create model
        model = SimplePulse(
            input_dim=64, state_dim=32, num_states=8, output_dim=10, gate_type="gru"
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Generate synthetic data
        batch_size = 8
        x = torch.randn(batch_size, 64).to(self.device)
        y_true = torch.randint(0, 10, (batch_size,)).to(self.device)

        # Forward pass
        output = model(x)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y_true)

        # Log metrics
        accuracy = (output.argmax(dim=1) == y_true).float().mean().item()
        logger.log_metrics({"loss": loss.item(), "accuracy": accuracy}, step=0)

        # Log state information (simplified for SimplePulse)
        logger.log_state_variables(
            model.initial_states, step=0, batch_idx=0, state_type="initial_states"
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ SimplePulse test completed")
        return logger.get_log_summary()

    def test_advanced_hybrid_model(self):
        """
        Test AdvancedHybridModel on synthetic data.
        """
        print("\n" + "=" * 50)
        print("Testing AdvancedHybridModel")
        print("=" * 50)

        # Create logger
        logger = PULSELogger(self.log_dir, "advanced_hybrid_test")

        # Create model
        config = {
            "input_dim": 64,
            "output_dim": 10,
            "embedding_dim": 32,
            "sequence_length": 4,
            "ssm_dim": 32,
            "ntm_mem_size": 32,
            "ntm_mem_dim": 16,
            "rnn_hidden_dim": 32,
            "attention_heads": 4,
        }

        model = AdvancedHybridModel(config).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Generate synthetic data
        batch_size = 8
        x = torch.randn(batch_size, 64).to(self.device)
        y_true = torch.randint(0, 10, (batch_size,)).to(self.device)

        # Forward pass
        output = model(x)

        # Get attention weights for logging
        attention_weights = model.get_attention_weights(x)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y_true)

        # Log metrics
        accuracy = (output.argmax(dim=1) == y_true).float().mean().item()
        logger.log_metrics({"loss": loss.item(), "accuracy": accuracy}, step=0)

        # Log attention weights
        logger.log_attention_weights(
            attention_weights, step=0, batch_idx=0, layer_name="self_attention"
        )

        # Log NTM memory content (simplified)
        # Note: In a real implementation, we would extract actual memory content
        dummy_memory = torch.randn(config["ntm_mem_size"], config["ntm_mem_dim"]).to(
            self.device
        )
        logger.log_memory_content(dummy_memory, step=0, batch_idx=0)

        # Log state variables
        logger.log_state_variables(
            output, step=0, batch_idx=0, state_type="final_output"
        )

        # Log batch data for visualization
        logger.log_batch_data(
            0, {"attention_weights": attention_weights, "memory_content": dummy_memory}
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Save visualizations
        logger.save_batch_visualizations(0)

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ AdvancedHybridModel test completed")
        return logger.get_log_summary()

    def test_sequential_hybrid_model(self):
        """
        Test SequentialHybridModel on synthetic data.
        """
        print("\n" + "=" * 50)
        print("Testing SequentialHybridModel")
        print("=" * 50)

        # Create logger
        logger = PULSELogger(self.log_dir, "sequential_hybrid_test")

        # Create model
        model = SequentialHybridModel(input_dim=64, output_dim=10).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Generate synthetic data
        batch_size = 8
        x = torch.randn(batch_size, 64).to(self.device)
        y_true = torch.randint(0, 10, (batch_size,)).to(self.device)

        # Forward pass
        output = model(x)

        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y_true)

        # Log metrics
        accuracy = (output.argmax(dim=1) == y_true).float().mean().item()
        logger.log_metrics({"loss": loss.item(), "accuracy": accuracy}, step=0)

        # Log state variables
        logger.log_state_variables(
            output, step=0, batch_idx=0, state_type="final_output"
        )

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Loss: {loss.item():.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("✓ SequentialHybridModel test completed")
        return logger.get_log_summary()

    def test_copy_task(self):
        """
        Test model on copy task synthetic data.
        """
        print("\n" + "=" * 50)
        print("Testing Copy Task")
        print("=" * 50)

        # Create logger
        logger = PULSELogger(self.log_dir, "copy_task_test")

        # Generate copy task data
        batch_size = 4
        seq_length = 5
        input_seq, target_seq = SyntheticDataGenerator.generate_copy_task(
            batch_size, seq_length
        )

        print(f"Input sequence shape: {input_seq.shape}")
        print(f"Target sequence shape: {target_seq.shape}")
        print(f"Sample input: {input_seq[0]}")
        print(f"Sample target: {target_seq[0]}")

        # Log the data
        logger.log_batch_data(
            0,
            {
                "input_sequence": input_seq[0].tolist(),
                "target_sequence": target_seq[0].tolist(),
            },
        )

        # For demonstration, we'll just log the sequences
        logger.log_metrics({"sequence_length": seq_length, "vocab_size": 10}, step=0)

        print("✓ Copy task test completed")
        return logger.get_log_summary()

    def run_all_tests(self):
        """
        Run all small-scale tests.
        """
        print("Running Small-Scale Tests on Synthetic Data")
        print("=" * 60)

        results = {}

        # Test SimplePulse
        try:
            results["simple_pulse"] = self.test_simple_pulse()
        except Exception as e:
            print(f"❌ SimplePulse test failed: {e}")
            results["simple_pulse"] = {"error": str(e)}

        # Test AdvancedHybridModel
        try:
            results["advanced_hybrid"] = self.test_advanced_hybrid_model()
        except Exception as e:
            print(f"❌ AdvancedHybridModel test failed: {e}")
            results["advanced_hybrid"] = {"error": str(e)}

        # Test SequentialHybridModel
        try:
            results["sequential_hybrid"] = self.test_sequential_hybrid_model()
        except Exception as e:
            print(f"❌ SequentialHybridModel test failed: {e}")
            results["sequential_hybrid"] = {"error": str(e)}

        # Test copy task
        try:
            results["copy_task"] = self.test_copy_task()
        except Exception as e:
            print(f"❌ Copy task test failed: {e}")
            results["copy_task"] = {"error": str(e)}

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for test_name, result in results.items():
            print(f"\n{test_name.upper()}:")
            if "error" in result:
                print(f"  ❌ Failed: {result['error']}")
            else:
                print(f"  ✅ Completed")
                for key, value in result.items():
                    print(f"    {key}: {value}")

        return results


# Example usage
if __name__ == "__main__":
    # Create test directory
    test_dir = "test_logs"
    os.makedirs(test_dir, exist_ok=True)

    # Run tests
    tester = SmallScaleTester(log_dir=test_dir)
    results = tester.run_all_tests()

    print(f"\n🎉 All tests completed! Logs saved to '{test_dir}' directory.")
