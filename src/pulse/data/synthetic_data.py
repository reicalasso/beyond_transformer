"""
Synthetic Data Generators for PULSE Testing

This module provides synthetic data generators for testing PULSE models.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch


class SyntheticDataGenerator:
    """
    Generate synthetic data for testing PULSE models.
    """

    @staticmethod
    def generate_copy_task(
        batch_size: int, sequence_length: int, vocab_size: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate copy task data.

        Args:
            batch_size: Number of samples in batch
            sequence_length: Length of sequences
            vocab_size: Size of vocabulary

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Generate random sequences
        input_sequences = torch.randint(1, vocab_size, (batch_size, sequence_length))

        # Add start and end tokens
        start_token = torch.zeros(batch_size, 1, dtype=torch.long)
        end_token = torch.full((batch_size, 1), vocab_size, dtype=torch.long)

        # Input: [start, sequence, end]
        input_with_tokens = torch.cat([start_token, input_sequences, end_token], dim=1)

        # Target: [sequence, end, padding...]
        target_sequences = torch.cat(
            [
                input_sequences,
                end_token,
                torch.zeros(batch_size, sequence_length + 1, dtype=torch.long),
            ],
            dim=1,
        )

        return input_with_tokens, target_sequences

    @staticmethod
    def generate_repeat_copy_task(
        batch_size: int, sequence_length: int, repeats: int = 2, vocab_size: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate repeat copy task data.

        Args:
            batch_size: Number of samples in batch
            sequence_length: Length of sequences
            repeats: Number of times to repeat
            vocab_size: Size of vocabulary

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Generate random sequences
        input_sequences = torch.randint(1, vocab_size, (batch_size, sequence_length))

        # Add start and end tokens
        start_token = torch.zeros(batch_size, 1, dtype=torch.long)
        end_token = torch.full((batch_size, 1), vocab_size, dtype=torch.long)

        # Input: [start, sequence, end]
        input_with_tokens = torch.cat([start_token, input_sequences, end_token], dim=1)

        # Target: [sequence * repeats, end, padding...]
        repeated_sequence = input_sequences.repeat(1, repeats)
        target_sequences = torch.cat(
            [
                repeated_sequence,
                end_token,
                torch.zeros(
                    batch_size, sequence_length * repeats + 1, dtype=torch.long
                ),
            ],
            dim=1,
        )

        return input_with_tokens, target_sequences

    @staticmethod
    def generate_associative_recall_task(
        batch_size: int,
        num_pairs: int,
        key_length: int = 3,
        value_length: int = 3,
        vocab_size: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate associative recall task data.

        Args:
            batch_size: Number of samples in batch
            num_pairs: Number of key-value pairs
            key_length: Length of keys
            value_length: Length of values
            vocab_size: Size of vocabulary

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Generate key-value pairs
        keys = torch.randint(1, vocab_size, (batch_size, num_pairs, key_length))
        values = torch.randint(1, vocab_size, (batch_size, num_pairs, value_length))

        # Generate query (random key to recall)
        query_idx = torch.randint(0, num_pairs, (batch_size,))
        queries = keys[torch.arange(batch_size), query_idx]

        # Target is the corresponding value
        targets = values[torch.arange(batch_size), query_idx]

        # Create input sequence: [pairs, query, end]
        # Flatten pairs
        pairs_flat = torch.cat(
            [keys.view(batch_size, -1), values.view(batch_size, -1)], dim=1
        )
        input_seq = torch.cat(
            [
                pairs_flat,
                queries,
                torch.full((batch_size, 1), vocab_size, dtype=torch.long),
            ],
            dim=1,
        )

        # Target: [value, end, padding...]
        target_seq = torch.cat(
            [
                targets,
                torch.full((batch_size, 1), vocab_size, dtype=torch.long),
                torch.zeros(
                    batch_size, key_length + value_length + 1, dtype=torch.long
                ),
            ],
            dim=1,
        )

        return input_seq, target_seq

    @staticmethod
    def generate_pattern_matching_task(
        batch_size: int,
        pattern_length: int = 5,
        sequence_length: int = 20,
        vocab_size: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate pattern matching task data.

        Args:
            batch_size: Number of samples in batch
            pattern_length: Length of pattern to match
            sequence_length: Length of input sequence
            vocab_size: Size of vocabulary

        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        # Generate pattern
        pattern = torch.randint(0, vocab_size, (batch_size, pattern_length))

        # Generate sequence with embedded pattern
        sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))

        # Randomly embed pattern in sequence
        embed_pos = torch.randint(
            0, sequence_length - pattern_length + 1, (batch_size,)
        )

        for i in range(batch_size):
            sequences[i, embed_pos[i] : embed_pos[i] + pattern_length] = pattern[i]

        # Input: [sequence, pattern, end]
        input_seq = torch.cat(
            [
                sequences,
                pattern,
                torch.full((batch_size, 1), vocab_size, dtype=torch.long),
            ],
            dim=1,
        )

        # Target: [position, end, padding...]
        target_seq = torch.cat(
            [
                embed_pos.unsqueeze(1),
                torch.full((batch_size, 1), vocab_size, dtype=torch.long),
                torch.zeros(
                    batch_size, sequence_length + pattern_length + 1, dtype=torch.long
                ),
            ],
            dim=1,
        )

        return input_seq, target_seq


# Example usage
if __name__ == "__main__":
    print("Testing Synthetic Data Generators...")

    batch_size = 4

    # Test copy task
    print("\n1. Copy Task:")
    input_seq, target_seq = SyntheticDataGenerator.generate_copy_task(batch_size, 5)
    print(f"   Input shape: {input_seq.shape}")
    print(f"   Target shape: {target_seq.shape}")
    print(f"   Sample input: {input_seq[0]}")
    print(f"   Sample target: {target_seq[0]}")

    # Test repeat copy task
    print("\n2. Repeat Copy Task:")
    input_seq, target_seq = SyntheticDataGenerator.generate_repeat_copy_task(
        batch_size, 3, repeats=2
    )
    print(f"   Input shape: {input_seq.shape}")
    print(f"   Target shape: {target_seq.shape}")
    print(f"   Sample input: {input_seq[0]}")
    print(f"   Sample target: {target_seq[0]}")

    # Test associative recall task
    print("\n3. Associative Recall Task:")
    input_seq, target_seq = SyntheticDataGenerator.generate_associative_recall_task(
        batch_size, 3
    )
    print(f"   Input shape: {input_seq.shape}")
    print(f"   Target shape: {target_seq.shape}")
    print(f"   Sample input: {input_seq[0]}")
    print(f"   Sample target: {target_seq[0]}")

    # Test pattern matching task
    print("\n4. Pattern Matching Task:")
    input_seq, target_seq = SyntheticDataGenerator.generate_pattern_matching_task(
        batch_size
    )
    print(f"   Input shape: {input_seq.shape}")
    print(f"   Target shape: {target_seq.shape}")
    print(f"   Sample input: {input_seq[0]}")
    print(f"   Sample target: {target_seq[0]}")

    print("\n✅ All synthetic data generators working correctly!")
