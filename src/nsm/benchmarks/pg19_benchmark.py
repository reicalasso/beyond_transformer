"""
PG-19 Benchmark for Neural State Machine Models

This module implements PG-19 benchmark tests for evaluating long-term memory capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Any
import random
import string


class PG19Dataset(Dataset):
    """
    PG-19-like dataset for long-term memory testing.
    """
    
    def __init__(self, size: int = 1000, max_length: int = 4096, 
                 vocab_size: int = 10000):
        """
        Initialize PG-19-like dataset.
        
        Args:
            size: Number of samples
            max_length: Maximum sequence length (simulating long documents)
            vocab_size: Vocabulary size
        """
        self.size = size
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Generate synthetic long-text data
        self.data = self._generate_pg19_data()
    
    def _generate_pg19_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic PG-19-like data.
        
        Returns:
            Tuple of (texts, targets)
        """
        texts = []
        targets = []
        
        for _ in range(self.size):
            # Generate long text sequence
            max_len = max(1001, min(8000, self.max_length))  # Ensure minimum length
            length = np.random.randint(1000, max_len)
            
            # Generate text with some structure (not completely random)
            # Simulate book-like text with paragraphs, chapters, etc.
            
            # Create vocabulary with some common words
            common_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'you', 'that', 'it',
                          'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i',
                          'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word',
                          'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said']
            
            # Generate text with some common words and random words
            text = []
            for _ in range(length):
                if np.random.random() < 0.3:  # 30% common words
                    text.append(np.random.choice(common_words))
                else:  # 70% random words
                    # Generate random word-like strings
                    word_length = np.random.randint(1, 10)
                    word = ''.join(np.random.choice(list(string.ascii_lowercase), word_length))
                    text.append(word)
            
            # Convert to indices
            # For synthetic data, we'll just use random indices
            text_indices = np.random.randint(0, self.vocab_size, length).tolist()
            
            # Pad or truncate to max_length
            if len(text_indices) > self.max_length:
                # Take a random segment for training
                start_idx = np.random.randint(0, len(text_indices) - self.max_length + 1)
                text_indices = text_indices[start_idx:start_idx + self.max_length]
            else:
                # Pad with zeros
                text_indices.extend([0] * (self.max_length - len(text_indices)))
            
            texts.append(text_indices)
            
            # Target: Next word prediction (simplified)
            # For demo, we'll use a simple classification task
            targets.append(np.random.randint(0, self.vocab_size))
        
        return torch.tensor(texts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.data[0][idx], self.data[1][idx]


class PG19Benchmark:
    """
    PG-19 Benchmark for Neural State Machine Models.
    """
    
    def __init__(self, model, device: torch.device = None):
        """
        Initialize PG-19 benchmark.
        
        Args:
            model: NSM model to benchmark
            device: Device to run benchmark on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def run_language_modeling(self, batch_size: int = 8, 
                            num_samples: int = 1000, 
                            seq_length: int = 2048) -> Dict[str, float]:
        """
        Run PG-19 language modeling benchmark.
        
        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples to test
            seq_length: Sequence length for processing
            
        Returns:
            Dictionary with benchmark results
        """
        print("Running PG-19 language modeling benchmark...")
        
        # Create dataset with long sequences
        dataset = PG19Dataset(size=num_samples, max_length=seq_length, vocab_size=5000)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation for language modeling
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_predictions = 0
        total_predictions = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (texts, targets) in enumerate(dataloader):
                texts, targets = texts.to(self.device), targets.to(self.device)
                
                # For long sequence processing, we might need to process in chunks
                # or use a model that can handle long sequences efficiently
                
                # Simplified approach: use average embedding
                if texts.dtype == torch.long:
                    # Create embedding layer for demonstration
                    embedding = nn.Embedding(5000, 128).to(self.device)
                    embedded = embedding(texts)  # [batch_size, seq_len, 128]
                else:
                    embedded = texts.unsqueeze(-1).repeat(1, 1, 128)  # [batch_size, seq_len, 128]
                
                # Global average pooling to handle variable lengths
                pooled = embedded.mean(dim=1)  # [batch_size, 128]
                
                # Forward pass
                outputs = self.model(pooled)
                
                # Ensure outputs match target dimension
                if outputs.size(-1) != 5000:  # vocab_size
                    projection = nn.Linear(outputs.size(-1), 5000).to(self.device)
                    outputs = projection(outputs)
                
                # Calculate loss (simplified target prediction)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)
                total_tokens += texts.size(1) * texts.size(0)  # Approximate token count
                
                # Limit for quick testing
                if batch_idx > 5:  # Just for demonstration
                    break
        
        avg_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        results = {
            'task': 'language_modeling',
            'perplexity': perplexity,
            'loss': avg_loss,
            'accuracy': accuracy,
            'tokens_processed': total_tokens,
            'samples_processed': total_predictions
        }
        
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Tokens: {total_tokens}")
        
        return results
    
    def run_long_context_memory(self, batch_size: int = 4, 
                               num_samples: int = 100,
                               context_length: int = 4096) -> Dict[str, float]:
        """
        Run long-context memory benchmark.
        
        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples to test
            context_length: Length of context to test memory on
            
        Returns:
            Dictionary with benchmark results
        """
        print("Running long-context memory benchmark...")
        
        # Create dataset with very long sequences
        dataset = PG19Dataset(size=num_samples, max_length=context_length, vocab_size=3000)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation for memory tasks
        self.model.eval()
        memory_scores = []
        
        with torch.no_grad():
            for batch_idx, (texts, targets) in enumerate(dataloader):
                texts, targets = texts.to(self.device), targets.to(self.device)
                
                # Simulate memory task: can model remember information from early in sequence?
                
                # Split sequence into segments
                segment_length = context_length // 4
                segments = texts.view(texts.size(0), 4, segment_length)
                
                # Simple memory test: check if model can associate first and last segments
                # This is a simplified version - real test would be more sophisticated
                
                # For demonstration, we'll compute a simple memory score
                # based on how well the model processes long sequences
                if hasattr(self.model, 'process_long_sequence'):
                    # If model has specific long sequence processing
                    outputs = self.model.process_long_sequence(texts.float())
                else:
                    # Simplified approach
                    outputs = self.model(texts.float().mean(dim=1))  # Average of sequence
                
                # Memory score based on output variance (higher variance might indicate better memory)
                memory_score = torch.var(outputs).item()
                memory_scores.append(memory_score)
                
                # Limit for quick testing
                if batch_idx > 3:  # Just for demonstration
                    break
        
        avg_memory_score = np.mean(memory_scores) if memory_scores else 0.0
        
        results = {
            'task': 'long_context_memory',
            'memory_score': avg_memory_score,
            'context_length': context_length,
            'samples_processed': len(memory_scores) * batch_size
        }
        
        print(f"  Memory Score: {avg_memory_score:.4f}")
        print(f"  Context Length: {context_length}")
        print(f"  Samples: {len(memory_scores) * batch_size}")
        
        return results


# Example usage
if __name__ == "__main__":
    print("Testing PG-19 Benchmark...")
    
    # Create a simple model for testing
    class SimplePG19Model(nn.Module):
        def __init__(self, input_dim=128, output_dim=1000):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # x shape: [batch_size, features]
            x = self.relu(self.fc1(x))
            output = self.fc2(x)  # [batch_size, output_dim]
            return output
    
    # Test with simple model
    model = SimplePG19Model()
    benchmark = PG19Benchmark(model)
    
    # Run language modeling test
    lm_results = benchmark.run_language_modeling(batch_size=4, num_samples=50, seq_length=1024)
    print(f"Language modeling results: {lm_results}")
    
    # Run memory test
    memory_results = benchmark.run_long_context_memory(batch_size=2, num_samples=20, context_length=2048)
    print(f"Memory results: {memory_results}")
    
    print("✅ PG-19 Benchmark test completed!")