"""
NTM (Neural Turing Machine) Memory Module Implementation

This module implements an NTM memory module that can be used
as a component in the PULSE architecture.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTMMemory(nn.Module):
    """
    NTM Memory Module.

    This module implements the memory component of a Neural Turing Machine,
    including read and write operations with content-based addressing.
    """

    def __init__(
        self,
        mem_size: int = 128,
        mem_dim: int = 20,
        num_read_heads: int = 1,
        num_write_heads: int = 1,
    ):
        """
        Initialize the NTMMemory.

        Args:
            mem_size: Number of memory slots
            mem_dim: Dimension of each memory slot
            num_read_heads: Number of read heads
            num_write_heads: Number of write heads
        """
        super(NTMMemory, self).__init__()

        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads

        # Initialize memory matrix
        self.register_buffer("memory", torch.randn(mem_size, mem_dim) * 0.01)

        # Initialize memory to zeros
        self.memory.fill_(0)

        # Create initial weight vectors for read and write heads
        self.register_buffer("read_weights", torch.zeros(num_read_heads, mem_size))
        self.register_buffer("write_weights", torch.zeros(num_write_heads, mem_size))

        # Initialize weights to uniform distribution
        self.read_weights.fill_(1.0 / mem_size)
        self.write_weights.fill_(1.0 / mem_size)

    def reset_memory(self):
        """Reset memory to initial state."""
        self.memory.fill_(0)
        self.read_weights.fill_(1.0 / self.mem_size)
        self.write_weights.fill_(1.0 / self.mem_size)

    def forward(
        self,
        read_keys: torch.Tensor,
        write_keys: torch.Tensor,
        read_strengths: torch.Tensor,
        write_strengths: torch.Tensor,
        erase_vectors: torch.Tensor,
        add_vectors: torch.Tensor,
        shift_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the NTM memory.

        Args:
            read_keys: Read head keys [batch_size, num_read_heads, mem_dim]
            write_keys: Write head keys [batch_size, num_write_heads, mem_dim]
            read_strengths: Read head strengths [batch_size, num_read_heads]
            write_strengths: Write head strengths [batch_size, num_write_heads]
            erase_vectors: Erase vectors [batch_size, num_write_heads, mem_dim]
            add_vectors: Add vectors [batch_size, num_write_heads, mem_dim]
            shift_weights: Shift weights for location-based addressing [batch_size, num_write_heads, shift_range]

        Returns:
            Tuple of (read_vectors, updated_memory)
            - read_vectors: [batch_size, num_read_heads, mem_dim]
            - updated_memory: [mem_size, mem_dim]
        """
        batch_size = read_keys.size(0)

        # Update write weights and memory
        # Use clone to allow gradient flow through memory operations
        updated_memory = self.memory.clone()

        for i in range(self.num_write_heads):
            # Content-based addressing for write head
            write_content_weights = self._content_based_addressing(
                write_keys[:, i, :], write_strengths[:, i]
            )

            # Location-based addressing (simplified)
            if shift_weights is not None and shift_weights.size(2) > 1:
                write_weights = self._location_based_addressing(
                    write_content_weights, shift_weights[:, i, :]
                )
            else:
                write_weights = write_content_weights

            # Update weights buffer (non-differentiable tracking)
            with torch.no_grad():
                self.write_weights[i] = write_weights.mean(dim=0)

            # Write operation - VECTORIZED for performance
            erase_vector = torch.sigmoid(
                erase_vectors[:, i, :]
            )  # [batch_size, mem_dim]
            add_vector = torch.tanh(add_vectors[:, i, :])  # [batch_size, mem_dim]

            # Vectorized write operation using einsum
            # write_weights: [batch_size, mem_size]
            # erase_vector: [batch_size, mem_dim]
            # Result: [batch_size, mem_size, mem_dim]
            erase_term = torch.einsum('bm,bd->bmd', write_weights, erase_vector)
            add_term = torch.einsum('bm,bd->bmd', write_weights, add_vector)

            # Apply erase and add operations (averaged across batch)
            # This maintains gradient flow while being vectorized
            avg_erase = erase_term.mean(dim=0)  # [mem_size, mem_dim]
            avg_add = add_term.mean(dim=0)  # [mem_size, mem_dim]
            updated_memory = updated_memory * (1 - avg_erase) + avg_add

        # Update memory buffer - keep gradient flow for training
        # Store detached version in buffer for next forward pass
        self.memory.data.copy_(updated_memory.detach().data)

        # Read operation
        read_vectors = []
        for i in range(self.num_read_heads):
            # Content-based addressing for read head
            read_content_weights = self._content_based_addressing(
                read_keys[:, i, :], read_strengths[:, i]
            )

            # Location-based addressing (simplified)
            if shift_weights is not None and shift_weights.size(2) > 1:
                read_weights = self._location_based_addressing(
                    read_content_weights, shift_weights[:, i, :]
                )
            else:
                read_weights = read_content_weights

            # Update weights
            self.read_weights[i] = read_weights.mean(dim=0)  # Average across batch

            # Read from memory
            read_vector = torch.matmul(
                read_weights, self.memory
            )  # [batch_size, mem_dim]
            read_vectors.append(read_vector)

        read_vectors = torch.stack(
            read_vectors, dim=1
        )  # [batch_size, num_read_heads, mem_dim]

        return read_vectors, self.memory

    def _content_based_addressing(
        self, keys: torch.Tensor, strengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Content-based addressing mechanism.

        Args:
            keys: Keys for addressing [batch_size, mem_dim]
            strengths: Addressing strengths [batch_size]

        Returns:
            Content weights [batch_size, mem_size]
        """
        # Normalize keys and memory for cosine similarity
        normalized_keys = F.normalize(keys, p=2, dim=-1)  # [batch_size, mem_dim]
        normalized_memory = F.normalize(self.memory, p=2, dim=-1)  # [mem_size, mem_dim]

        # Compute cosine similarity
        similarity = torch.matmul(
            normalized_keys, normalized_memory.t()
        )  # [batch_size, mem_size]

        # Apply softmax with strength
        strengths = F.softplus(strengths).unsqueeze(-1)  # [batch_size, 1]
        weighted_similarity = similarity * strengths  # [batch_size, mem_size]
        content_weights = F.softmax(
            weighted_similarity, dim=-1
        )  # [batch_size, mem_size]

        return content_weights

    def _location_based_addressing(
        self, content_weights: torch.Tensor, shift_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Location-based addressing mechanism.

        Args:
            content_weights: Content-based weights [batch_size, mem_size]
            shift_weights: Shift weights [batch_size, shift_range]

        Returns:
            Location-based weights [batch_size, mem_size]
        """
        batch_size = content_weights.size(0)
        shift_range = shift_weights.size(1)

        # Create circular convolution matrix
        conv_matrix = self._create_circulant_matrix(shift_range)

        # Apply shift
        shifted_weights = torch.matmul(
            shift_weights.unsqueeze(1), conv_matrix.expand(batch_size, -1, -1)
        )
        shifted_weights = shifted_weights.squeeze(1)  # [batch_size, mem_size]

        # Interpolate between content and shifted weights
        # This is a simplified version - in practice, you'd have an interpolation gate
        interpolated_weights = (content_weights + shifted_weights) / 2

        return F.softmax(interpolated_weights, dim=-1)

    def _create_circulant_matrix(self, shift_range: int) -> torch.Tensor:
        """
        Create circulant matrix for shift operation.

        Args:
            shift_range: Range of shifts

        Returns:
            Circulant matrix [mem_size, mem_size]
        """
        # Create identity matrix
        eye = torch.eye(self.mem_size)

        # Create shifts (simplified)
        shifts = torch.arange(shift_range) - shift_range // 2

        # Create circulant matrix (this is a simplified version)
        circulant = torch.zeros(self.mem_size, self.mem_size)
        for i in range(self.mem_size):
            circulant[i, (i + shifts[shift_range // 2]) % self.mem_size] = 1

        return circulant

    def get_memory_state(self) -> torch.Tensor:
        """
        Get current memory state.

        Returns:
            Memory matrix [mem_size, mem_dim]
        """
        return self.memory.clone()

    def get_read_weights(self) -> torch.Tensor:
        """
        Get current read weights.

        Returns:
            Read weights [num_read_heads, mem_size]
        """
        return self.read_weights.clone()

    def get_write_weights(self) -> torch.Tensor:
        """
        Get current write weights.

        Returns:
            Write weights [num_write_heads, mem_size]
        """
        return self.write_weights.clone()


# Example usage
if __name__ == "__main__":
    # Test NTMMemory
    batch_size, mem_size, mem_dim = 2, 128, 20
    num_read_heads, num_write_heads = 1, 1

    # Create NTMMemory
    ntm_memory = NTMMemory(
        mem_size=mem_size,
        mem_dim=mem_dim,
        num_read_heads=num_read_heads,
        num_write_heads=num_write_heads,
    )

    # Create sample inputs
    read_keys = torch.randn(batch_size, num_read_heads, mem_dim)
    write_keys = torch.randn(batch_size, num_write_heads, mem_dim)
    read_strengths = torch.randn(batch_size, num_read_heads)
    write_strengths = torch.randn(batch_size, num_write_heads)
    erase_vectors = torch.randn(batch_size, num_write_heads, mem_dim)
    add_vectors = torch.randn(batch_size, num_write_heads, mem_dim)

    # Forward pass
    read_vectors, memory_state = ntm_memory(
        read_keys,
        write_keys,
        read_strengths,
        write_strengths,
        erase_vectors,
        add_vectors,
    )

    print(f"Read vectors shape: {read_vectors.shape}")
    print(f"Memory state shape: {memory_state.shape}")
    print("NTMMemory test completed successfully!")
