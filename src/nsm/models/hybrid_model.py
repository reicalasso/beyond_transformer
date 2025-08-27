"""
Advanced Hybrid Neural State Machine Model

This module implements an advanced hybrid model that properly integrates SSM, NTM, Transformer Attention, 
and RNN components in a sequential flow with proper data transformations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Import our core components
from nsm.modules.ssm_block import SSMBlock
from nsm.modules.ntm_memory import NTMMemory
from nsm.modules.transformer_attention import TransformerAttention
from nsm.modules.rnn_memory import RNNMemory


class AdvancedHybridModel(nn.Module):
    """
    Advanced Hybrid Model integrating SSM, NTM, Transformer Attention, and RNN components.
    
    Data Flow: Input → Embedding → Attention Processing → SSM Processing → 
               NTM Memory Operations → RNN Processing → Output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AdvancedHybridModel.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(AdvancedHybridModel, self).__init__()
        
        # Default configuration
        default_config = {
            'input_dim': 784,
            'sequence_length': 16,
            'embedding_dim': 128,
            'attention_heads': 8,
            'ssm_dim': 128,
            'ssm_state_dim': 16,
            'ssm_conv_dim': 4,
            'ntm_mem_size': 128,
            'ntm_mem_dim': 20,
            'ntm_read_heads': 1,
            'ntm_write_heads': 1,
            'rnn_hidden_dim': 128,
            'rnn_layers': 2,
            'rnn_type': 'lstm',
            'output_dim': 10,
            'dropout': 0.1
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # Extract configuration
        self.input_dim = default_config['input_dim']
        self.seq_len = default_config['sequence_length']
        self.embedding_dim = default_config['embedding_dim']
        self.output_dim = default_config['output_dim']
        
        # Input processing
        self.input_projection = nn.Linear(self.input_dim, self.embedding_dim * self.seq_len)
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, self.embedding_dim))
        
        # Layer normalization
        self.embedding_norm = nn.LayerNorm(self.embedding_dim)
        self.attention_norm = nn.LayerNorm(self.embedding_dim)
        self.ssm_norm = nn.LayerNorm(self.embedding_dim)
        
        # Transformer Attention
        self.attention = TransformerAttention(
            d_model=self.embedding_dim,
            num_heads=default_config['attention_heads'],
            dropout=default_config['dropout']
        )
        
        # SSM Block
        self.ssm_block = SSMBlock(
            d_model=self.embedding_dim,
            d_state=default_config['ssm_state_dim'],
            d_conv=default_config['ssm_conv_dim'],
            expand=default_config['ssm_dim'] // self.embedding_dim if default_config['ssm_dim'] > self.embedding_dim else 2
        )
        
        # NTM Memory Components
        self.ntm_memory = NTMMemory(
            mem_size=default_config['ntm_mem_size'],
            mem_dim=default_config['ntm_mem_dim'],
            num_read_heads=default_config['ntm_read_heads'],
            num_write_heads=default_config['ntm_write_heads']
        )
        
        # NTM parameter generators
        ntm_mem_dim = default_config['ntm_mem_dim']
        read_heads = default_config['ntm_read_heads']
        write_heads = default_config['ntm_write_heads']
        
        # Read head parameter generators
        self.read_key_generator = nn.Linear(self.embedding_dim, read_heads * ntm_mem_dim)
        self.read_strength_generator = nn.Linear(self.embedding_dim, read_heads)
        
        # Write head parameter generators
        self.write_key_generator = nn.Linear(self.embedding_dim, write_heads * ntm_mem_dim)
        self.write_strength_generator = nn.Linear(self.embedding_dim, write_heads)
        self.erase_vector_generator = nn.Linear(self.embedding_dim, write_heads * ntm_mem_dim)
        self.add_vector_generator = nn.Linear(self.embedding_dim, write_heads * ntm_mem_dim)
        
        # RNN Memory
        rnn_input_dim = self.embedding_dim + (read_heads * ntm_mem_dim)  # Attention output + NTM read vectors
        self.rnn_memory = RNNMemory(
            input_dim=rnn_input_dim,
            hidden_dim=default_config['rnn_hidden_dim'],
            num_layers=default_config['rnn_layers'],
            rnn_type=default_config['rnn_type'],
            dropout=default_config['dropout']
        )
        
        # Output layers
        self.output_projection = nn.Linear(default_config['rnn_hidden_dim'], self.output_dim)
        self.output_dropout = nn.Dropout(default_config['dropout'])
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdvancedHybridModel.
        
        Data Flow: Input → Embedding → Attention → SSM → NTM → RNN → Output
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # 1. Input Processing
        # Project input to sequence
        embedded_seq = self.input_projection(x)  # [batch_size, embedding_dim * seq_len]
        embedded_seq = embedded_seq.view(batch_size, self.seq_len, self.embedding_dim)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        embedded_seq = embedded_seq + self.positional_encoding.unsqueeze(0)  # [batch_size, seq_len, embedding_dim]
        embedded_seq = self.embedding_norm(embedded_seq)
        
        # 2. Attention Processing
        attended_seq, attention_weights = self.attention.forward_self_attention(embedded_seq)
        attended_seq = self.attention_norm(attended_seq)
        
        # Use mean pooling of attended sequence
        attention_features = attended_seq.mean(dim=1)  # [batch_size, embedding_dim]
        
        # 3. SSM Processing
        ssm_output = self.ssm_block(attended_seq)  # [batch_size, seq_len, embedding_dim]
        ssm_output = self.ssm_norm(ssm_output)
        
        # Use the last time step from SSM
        ssm_features = ssm_output[:, -1, :]  # [batch_size, embedding_dim]
        
        # 4. NTM Memory Operations
        # Generate NTM parameters from SSM features
        read_keys = self.read_key_generator(ssm_features).view(batch_size, self.config['ntm_read_heads'], self.config['ntm_mem_dim'])
        read_strengths = F.softplus(self.read_strength_generator(ssm_features))
        
        write_keys = self.write_key_generator(ssm_features).view(batch_size, self.config['ntm_write_heads'], self.config['ntm_mem_dim'])
        write_strengths = F.softplus(self.write_strength_generator(ssm_features))
        erase_vectors = torch.sigmoid(self.erase_vector_generator(ssm_features)).view(batch_size, self.config['ntm_write_heads'], self.config['ntm_mem_dim'])
        add_vectors = torch.tanh(self.add_vector_generator(ssm_features)).view(batch_size, self.config['ntm_write_heads'], self.config['ntm_mem_dim'])
        
        # NTM memory operations
        read_vectors, _ = self.ntm_memory(
            read_keys, write_keys, read_strengths, write_strengths,
            erase_vectors, add_vectors
        )  # [batch_size, num_read_heads, ntm_mem_dim]
        
        # Flatten read vectors
        read_vectors_flat = read_vectors.view(batch_size, -1)  # [batch_size, num_read_heads * ntm_mem_dim]
        
        # 5. RNN Processing
        # Combine attention features with NTM read vectors
        rnn_input_features = torch.cat([attention_features, read_vectors_flat], dim=-1)  # [batch_size, embedding_dim + num_read_heads * ntm_mem_dim]
        
        # Expand to sequence for RNN processing
        rnn_input_seq = rnn_input_features.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, ...]
        
        # Initialize RNN hidden state
        device = rnn_input_seq.device
        hidden = self.rnn_memory.init_hidden(batch_size, device)
        
        # RNN processing
        rnn_output_seq, _ = self.rnn_memory(rnn_input_seq, hidden)  # [batch_size, seq_len, rnn_hidden_dim]
        
        # Use the last time step from RNN
        rnn_features = rnn_output_seq[:, -1, :]  # [batch_size, rnn_hidden_dim]
        
        # 6. Output Projection
        output = self.output_projection(rnn_features)  # [batch_size, output_dim]
        output = self.output_dropout(output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Attention weights of shape [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = x.shape[0]
        
        # Process input through embedding
        embedded_seq = self.input_projection(x)
        embedded_seq = embedded_seq.view(batch_size, self.seq_len, self.embedding_dim)
        embedded_seq = embedded_seq + self.positional_encoding.unsqueeze(0)
        embedded_seq = self.embedding_norm(embedded_seq)
        
        # Get attention weights
        _, attention_weights = self.attention.forward_self_attention(embedded_seq)
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'config': self.config,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_details': {
                'input_projection': sum(p.numel() for p in self.input_projection.parameters()),
                'attention': sum(p.numel() for p in self.attention.parameters()),
                'ssm_block': sum(p.numel() for p in self.ssm_block.parameters()),
                'ntm_memory': sum(p.numel() for p in self.ntm_memory.parameters()),
                'rnn_memory': sum(p.numel() for p in self.rnn_memory.parameters()),
                'output_projection': sum(p.numel() for p in self.output_projection.parameters())
            }
        }


# Simplified Hybrid Model with exact flow: Input → Attention → SSM → NTM → RNN → Output
class SequentialHybridModel(nn.Module):
    """
    Sequential Hybrid Model with exact component flow.
    
    Data Flow: Input → Attention → SSM → NTM → RNN → Output
    """
    
    def __init__(self, input_dim: int = 784, output_dim: int = 10):
        """
        Initialize the SequentialHybridModel.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(SequentialHybridModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = 128
        self.seq_len = 8
        
        # Input embedding to sequence
        self.input_to_sequence = nn.Linear(input_dim, self.embedding_dim * self.seq_len)
        self.positional_encoding = nn.Parameter(torch.randn(self.seq_len, self.embedding_dim))
        
        # Attention layer
        self.attention = TransformerAttention(d_model=self.embedding_dim, num_heads=8)
        
        # SSM layer
        self.ssm = SSMBlock(d_model=self.embedding_dim, d_state=16)
        
        # NTM layer
        self.ntm = NTMMemory(mem_size=64, mem_dim=16, num_read_heads=1, num_write_heads=1)
        
        # NTM parameter generators
        self.ntm_read_key = nn.Linear(self.embedding_dim, 16)
        self.ntm_write_key = nn.Linear(self.embedding_dim, 16)
        self.ntm_read_strength = nn.Linear(self.embedding_dim, 1)
        self.ntm_write_strength = nn.Linear(self.embedding_dim, 1)
        self.ntm_erase = nn.Linear(self.embedding_dim, 16)
        self.ntm_add = nn.Linear(self.embedding_dim, 16)
        
        # RNN layer
        self.rnn = RNNMemory(input_dim=self.embedding_dim + 16, hidden_dim=128, rnn_type='gru')
        
        # Output layer
        self.output_layer = nn.Linear(128, output_dim)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with exact flow: Input → Attention → SSM → NTM → RNN → Output
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        
        # Input → Sequence
        seq_x = self.input_to_sequence(x).view(batch_size, self.seq_len, self.embedding_dim)
        seq_x = seq_x + self.positional_encoding.unsqueeze(0)
        
        # Attention
        attended_x, _ = self.attention.forward_self_attention(seq_x)
        attention_feature = attended_x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # SSM
        ssm_output = self.ssm(seq_x)
        ssm_feature = ssm_output.mean(dim=1)  # [batch_size, embedding_dim]
        
        # NTM
        read_key = self.ntm_read_key(ssm_feature).unsqueeze(1)  # [batch_size, 1, 16]
        write_key = self.ntm_write_key(ssm_feature).unsqueeze(1)  # [batch_size, 1, 16]
        read_strength = F.softplus(self.ntm_read_strength(ssm_feature).squeeze(-1))  # [batch_size]
        write_strength = F.softplus(self.ntm_write_strength(ssm_feature).squeeze(-1))  # [batch_size]
        erase_vector = torch.sigmoid(self.ntm_erase(ssm_feature)).unsqueeze(1)  # [batch_size, 1, 16]
        add_vector = torch.tanh(self.ntm_add(ssm_feature)).unsqueeze(1)  # [batch_size, 1, 16]
        
        read_vectors, _ = self.ntm(
            read_key, write_key, 
            read_strength.unsqueeze(-1), write_strength.unsqueeze(-1),
            erase_vector, add_vector
        )
        ntm_feature = read_vectors.squeeze(1)  # [batch_size, 16]
        
        # RNN
        rnn_input = torch.cat([attention_feature, ntm_feature], dim=-1).unsqueeze(1)  # [batch_size, 1, embedding_dim+16]
        rnn_input = rnn_input.repeat(1, 4, 1)  # [batch_size, 4, embedding_dim+16]
        hidden = self.rnn.init_hidden(batch_size, rnn_input.device)
        rnn_output, _ = self.rnn(rnn_input, hidden)
        rnn_feature = rnn_output[:, -1, :]  # [batch_size, 128]
        
        # Output
        output = self.output_layer(rnn_feature)
        
        return output


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Models...")
    
    # Test AdvancedHybridModel
    print("\n1. Testing AdvancedHybridModel...")
    config = {
        'input_dim': 784,
        'output_dim': 10,
        'embedding_dim': 64,
        'sequence_length': 8
    }
    
    advanced_model = AdvancedHybridModel(config)
    info = advanced_model.get_model_info()
    print(f"   Model created with {info['total_parameters']:,} parameters")
    
    # Test with sample input
    batch_size = 2
    x = torch.randn(batch_size, 784)
    output = advanced_model(x)
    print(f"   Input: {x.shape} → Output: {output.shape}")
    
    # Test attention weights
    attn_weights = advanced_model.get_attention_weights(x)
    print(f"   Attention weights: {attn_weights.shape}")
    
    # Test SequentialHybridModel
    print("\n2. Testing SequentialHybridModel...")
    sequential_model = SequentialHybridModel(input_dim=784, output_dim=10)
    
    x = torch.randn(batch_size, 784)
    output = sequential_model(x)
    print(f"   Input: {x.shape} → Output: {output.shape}")
    
    # Test differentiability
    loss = output.sum()
    loss.backward()
    print("   ✓ Backward pass successful")
    
    print("\n🎉 All hybrid models tested successfully!")