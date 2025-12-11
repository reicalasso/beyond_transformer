"""
Adaptive State Management for PULSEs.

This module implements advanced state management mechanisms:
- AdaptiveStateAllocator: Dynamic state allocation based on input complexity
- StateCompressor: Efficient state compression for memory optimization
- HierarchicalStateManager: Multi-level state hierarchy for complex reasoning
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveStateAllocator(nn.Module):
    """
    Dynamically allocates states based on input complexity.
    
    Uses a learned complexity estimator to determine the optimal number
    of active states for each input, balancing efficiency and capacity.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        min_states: int = 4,
        max_states: int = 64,
        initial_states: int = 16,
    ) -> None:
        """
        Initialize AdaptiveStateAllocator.

        Args:
            input_dim: Dimension of input features.
            state_dim: Dimension of each state vector.
            min_states: Minimum number of active states.
            max_states: Maximum number of active states.
            initial_states: Initial number of states to allocate.
        """
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.min_states = min_states
        self.max_states = max_states
        self.initial_states = initial_states

        # State bank - all possible states
        self.state_bank = nn.Parameter(torch.randn(max_states, state_dim) * 0.02)

        # Complexity estimator network
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid(),
        )

        # State importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1),
        )

        # Learnable temperature for soft selection
        self.temperature = nn.Parameter(torch.ones(1))

    def estimate_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate input complexity to determine state count.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]

        Returns:
            Complexity scores [batch_size] in range [0, 1]
        """
        if x.dim() == 3:
            # Sequence input - use mean pooling
            x_pooled = x.mean(dim=1)
        else:
            x_pooled = x

        return self.complexity_estimator(x_pooled).squeeze(-1)

    def compute_num_states(self, complexity: torch.Tensor) -> torch.Tensor:
        """
        Compute number of states based on complexity.

        Args:
            complexity: Complexity scores [batch_size]

        Returns:
            Number of states for each sample [batch_size]
        """
        # Linear interpolation between min and max states
        num_states = self.min_states + complexity * (self.max_states - self.min_states)
        return num_states.round().long().clamp(self.min_states, self.max_states)

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Allocate states based on input complexity.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            return_info: If True, return allocation info dict

        Returns:
            Tuple of (allocated_states, state_mask)
            - allocated_states: [batch_size, max_states, state_dim]
            - state_mask: [batch_size, max_states] boolean mask of active states
        """
        batch_size = x.shape[0]

        # Estimate complexity
        complexity = self.estimate_complexity(x)

        # Compute importance scores for each state
        importance = self.importance_scorer(self.state_bank).squeeze(-1)  # [max_states]
        importance = F.softmax(importance / self.temperature.abs().clamp(min=0.1), dim=-1)

        # Compute number of states per sample
        num_states = self.compute_num_states(complexity)

        # Create state mask based on importance ranking
        # Sort states by importance and select top-k for each sample
        sorted_indices = torch.argsort(importance, descending=True)

        # Create masks for each sample
        state_mask = torch.zeros(batch_size, self.max_states, dtype=torch.bool, device=x.device)
        for i in range(batch_size):
            k = num_states[i].item()
            state_mask[i, sorted_indices[:k]] = True

        # Expand state bank for batch
        allocated_states = self.state_bank.unsqueeze(0).expand(batch_size, -1, -1)

        if return_info:
            info = {
                "complexity": complexity,
                "num_states": num_states,
                "importance": importance,
                "active_ratio": state_mask.float().mean(),
            }
            return allocated_states, state_mask, info

        return allocated_states, state_mask

    def get_active_states(
        self,
        allocated_states: torch.Tensor,
        state_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get only the active states for each sample.

        Args:
            allocated_states: [batch_size, max_states, state_dim]
            state_mask: [batch_size, max_states]

        Returns:
            List of active state tensors, one per sample
        """
        batch_size = allocated_states.shape[0]
        active_states = []

        for i in range(batch_size):
            mask = state_mask[i]
            active = allocated_states[i, mask]
            active_states.append(active)

        return active_states


class StateCompressor(nn.Module):
    """
    Compresses states for memory-efficient processing.
    
    Uses learned compression to reduce state dimensionality while
    preserving important information.
    """

    def __init__(
        self,
        state_dim: int,
        compressed_dim: int,
        num_codebook_entries: int = 256,
    ) -> None:
        """
        Initialize StateCompressor.

        Args:
            state_dim: Original state dimension.
            compressed_dim: Compressed state dimension.
            num_codebook_entries: Number of entries in quantization codebook.
        """
        super().__init__()
        self.state_dim = state_dim
        self.compressed_dim = compressed_dim
        self.num_codebook_entries = num_codebook_entries

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, compressed_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, state_dim),
        )

        # Optional: Vector quantization codebook
        self.codebook = nn.Parameter(torch.randn(num_codebook_entries, compressed_dim))

        # Commitment loss weight
        self.commitment_weight = 0.25

    def compress(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compress states to lower dimension.

        Args:
            states: State tensor [batch_size, num_states, state_dim]

        Returns:
            Compressed states [batch_size, num_states, compressed_dim]
        """
        return self.encoder(states)

    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """
        Decompress states back to original dimension.

        Args:
            compressed: Compressed tensor [batch_size, num_states, compressed_dim]

        Returns:
            Decompressed states [batch_size, num_states, state_dim]
        """
        return self.decoder(compressed)

    def quantize(
        self, compressed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize compressed states using codebook.

        Args:
            compressed: Compressed tensor [batch_size, num_states, compressed_dim]

        Returns:
            Tuple of (quantized, indices, commitment_loss)
        """
        batch_size, num_states, _ = compressed.shape

        # Flatten for distance computation
        flat = compressed.view(-1, self.compressed_dim)

        # Compute distances to codebook entries
        distances = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.pow(2).sum(dim=-1)
            - 2 * torch.matmul(flat, self.codebook.t())
        )

        # Find nearest codebook entries
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)

        # Reshape
        quantized = quantized.view(batch_size, num_states, self.compressed_dim)
        indices = indices.view(batch_size, num_states)

        # Straight-through estimator for gradients
        quantized = compressed + (quantized - compressed).detach()

        # Commitment loss
        commitment_loss = F.mse_loss(compressed, quantized.detach())

        return quantized, indices, commitment_loss * self.commitment_weight

    def forward(
        self, states: torch.Tensor, use_quantization: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compress and decompress states.

        Args:
            states: State tensor [batch_size, num_states, state_dim]
            use_quantization: Whether to apply vector quantization

        Returns:
            Tuple of (reconstructed_states, info_dict)
        """
        compressed = self.compress(states)

        if use_quantization:
            quantized, indices, commitment_loss = self.quantize(compressed)
            reconstructed = self.decompress(quantized)
            info = {
                "compressed": compressed,
                "quantized": quantized,
                "indices": indices,
                "commitment_loss": commitment_loss,
            }
        else:
            reconstructed = self.decompress(compressed)
            info = {"compressed": compressed}

        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, states)
        info["reconstruction_loss"] = reconstruction_loss

        return reconstructed, info


class HierarchicalStateManager(nn.Module):
    """
    Multi-level state hierarchy for complex reasoning.
    
    Maintains states at multiple levels of abstraction:
    - Token-level: Fine-grained, local information
    - Chunk-level: Medium-grained, phrase/sentence information
    - Global-level: Coarse-grained, document/context information
    """

    def __init__(
        self,
        state_dim: int,
        num_token_states: int = 32,
        num_chunk_states: int = 16,
        num_global_states: int = 8,
        chunk_size: int = 64,
    ) -> None:
        """
        Initialize HierarchicalStateManager.

        Args:
            state_dim: Dimension of state vectors.
            num_token_states: Number of token-level states.
            num_chunk_states: Number of chunk-level states.
            num_global_states: Number of global-level states.
            chunk_size: Size of chunks for chunk-level processing.
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_token_states = num_token_states
        self.num_chunk_states = num_chunk_states
        self.num_global_states = num_global_states
        self.chunk_size = chunk_size

        # State banks for each level
        self.token_states = nn.Parameter(torch.randn(num_token_states, state_dim) * 0.02)
        self.chunk_states = nn.Parameter(torch.randn(num_chunk_states, state_dim) * 0.02)
        self.global_states = nn.Parameter(torch.randn(num_global_states, state_dim) * 0.02)

        # Cross-level attention
        self.token_to_chunk = nn.MultiheadAttention(state_dim, num_heads=4, batch_first=True)
        self.chunk_to_global = nn.MultiheadAttention(state_dim, num_heads=4, batch_first=True)
        self.global_to_token = nn.MultiheadAttention(state_dim, num_heads=4, batch_first=True)

        # Update gates for each level
        self.token_gate = nn.Linear(state_dim * 2, state_dim)
        self.chunk_gate = nn.Linear(state_dim * 2, state_dim)
        self.global_gate = nn.Linear(state_dim * 2, state_dim)

        # Layer norms
        self.token_norm = nn.LayerNorm(state_dim)
        self.chunk_norm = nn.LayerNorm(state_dim)
        self.global_norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        x: torch.Tensor,
        prev_states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process input through hierarchical state levels.

        Args:
            x: Input tensor [batch_size, seq_len, state_dim]
            prev_states: Optional previous states from last step

        Returns:
            Tuple of (output, updated_states)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize or use previous states
        if prev_states is None:
            token_s = self.token_states.unsqueeze(0).expand(batch_size, -1, -1)
            chunk_s = self.chunk_states.unsqueeze(0).expand(batch_size, -1, -1)
            global_s = self.global_states.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            token_s = prev_states["token"]
            chunk_s = prev_states["chunk"]
            global_s = prev_states["global"]

        # Token-level update: tokens attend to token states
        token_update, _ = self.token_to_chunk(x, token_s, token_s)
        token_gate_input = torch.cat([token_s, token_update.mean(dim=1, keepdim=True).expand(-1, self.num_token_states, -1)], dim=-1)
        token_gate_weight = torch.sigmoid(self.token_gate(token_gate_input))
        token_s = self.token_norm(token_s + token_gate_weight * token_update.mean(dim=1, keepdim=True).expand(-1, self.num_token_states, -1))

        # Chunk-level update: chunk states attend to token states
        chunk_update, _ = self.chunk_to_global(chunk_s, token_s, token_s)
        chunk_gate_input = torch.cat([chunk_s, chunk_update], dim=-1)
        chunk_gate_weight = torch.sigmoid(self.chunk_gate(chunk_gate_input))
        chunk_s = self.chunk_norm(chunk_s + chunk_gate_weight * chunk_update)

        # Global-level update: global states attend to chunk states
        global_update, _ = self.chunk_to_global(global_s, chunk_s, chunk_s)
        global_gate_input = torch.cat([global_s, global_update], dim=-1)
        global_gate_weight = torch.sigmoid(self.global_gate(global_gate_input))
        global_s = self.global_norm(global_s + global_gate_weight * global_update)

        # Top-down influence: global informs token processing
        output, _ = self.global_to_token(x, global_s, global_s)
        output = x + output

        # Package updated states
        updated_states = {
            "token": token_s,
            "chunk": chunk_s,
            "global": global_s,
        }

        return output, updated_states

    def get_state_summary(
        self, states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get a summary vector from all state levels.

        Args:
            states: Dictionary of state tensors

        Returns:
            Summary tensor [batch_size, state_dim]
        """
        # Concatenate mean of each level and project
        token_mean = states["token"].mean(dim=1)
        chunk_mean = states["chunk"].mean(dim=1)
        global_mean = states["global"].mean(dim=1)

        # Weighted combination
        summary = (token_mean + chunk_mean + global_mean) / 3
        return summary
