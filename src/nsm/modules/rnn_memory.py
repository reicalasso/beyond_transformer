"""
RNN Memory Layer Implementation

This module implements an RNN-based memory layer that can be used
as a component in the Neural State Machine architecture.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RNNMemory(nn.Module):
    """
    RNN-based Memory Layer.

    This module implements an RNN memory layer that can use LSTM or GRU
    as the underlying recurrent unit.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        """
        Initialize the RNNMemory.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN ('lstm', 'gru', or 'rnn')
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
        """
        super(RNNMemory, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Create RNN layer
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: {rnn_type}. Use 'lstm', 'gru', or 'rnn'."
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim * self.num_directions, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the RNNMemory.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            hidden: Initial hidden state (h0) or (h0, c0) for LSTM

        Returns:
            Tuple of (output, hidden_state)
            - output: [batch_size, seq_len, hidden_dim]
            - hidden_state: Final hidden state (h_n) or (h_n, c_n) for LSTM
        """
        # RNN forward pass
        rnn_output, hidden_state = self.rnn(x, hidden)

        # Apply output projection
        output = self.output_proj(rnn_output)
        output = self.dropout(output)

        return output, hidden_state

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size
            device: Device to initialize on

        Returns:
            Initial hidden state
        """
        h0 = torch.zeros(
            self.num_layers * self.num_directions, batch_size, self.hidden_dim
        ).to(device)

        if self.rnn_type == "lstm":
            c0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_dim
            ).to(device)
            return (h0, c0)
        else:
            return h0

    def forward_step(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for a single time step.

        Args:
            x: Input tensor [batch_size, input_dim]
            hidden: Initial hidden state

        Returns:
            Tuple of (output, hidden_state)
        """
        # Add sequence dimension
        x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Forward pass
        output, hidden_state = self.forward(x, hidden)

        # Remove sequence dimension
        output = output.squeeze(1)  # [batch_size, hidden_dim]

        return output, hidden_state


# Example usage
if __name__ == "__main__":
    # Test RNNMemory
    batch_size, seq_len, input_dim, hidden_dim = 2, 10, 64, 128
    num_layers = 2

    # Test LSTM
    print("Testing LSTM...")
    lstm_memory = RNNMemory(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="lstm",
    )

    # Create sample input
    x = torch.randn(batch_size, seq_len, input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    lstm_memory = lstm_memory.to(device)

    # Initialize hidden state
    hidden = lstm_memory.init_hidden(batch_size, device)

    # Forward pass
    output, hidden_state = lstm_memory(x, hidden)

    print(f"LSTM - Input shape: {x.shape}")
    print(f"LSTM - Output shape: {output.shape}")
    print(f"LSTM - Hidden state shape: {hidden_state[0].shape}")

    # Test GRU
    print("\nTesting GRU...")
    gru_memory = RNNMemory(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type="gru",
    )
    gru_memory = gru_memory.to(device)

    # Initialize hidden state
    hidden = gru_memory.init_hidden(batch_size, device)

    # Forward pass
    output, hidden_state = gru_memory(x, hidden)

    print(f"GRU - Input shape: {x.shape}")
    print(f"GRU - Output shape: {output.shape}")
    print(f"GRU - Hidden state shape: {hidden_state.shape}")

    # Test single step
    print("\nTesting single step...")
    single_input = torch.randn(batch_size, input_dim).to(device)
    single_output, single_hidden = gru_memory.forward_step(single_input, hidden)
    print(f"Single step - Input shape: {single_input.shape}")
    print(f"Single step - Output shape: {single_output.shape}")

    print("RNNMemory test completed successfully!")
