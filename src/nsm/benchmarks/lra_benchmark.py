"""
LRA (Long Range Arena) Benchmark for Neural State Machine Models

This module implements LRA benchmark tests for NSM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, Any
import tempfile
import os


class LRADataset(Dataset):
    """
    Synthetic LRA dataset for benchmarking.
    """
    
    def __init__(self, task_type: str = "listops", size: int = 1000, 
                 max_length: int = 1000):
        """
        Initialize LRA dataset.
        
        Args:
            task_type: Type of LRA task ("listops", "text", "retrieval", "image", "pathfinder")
            size: Number of samples
            max_length: Maximum sequence length
        """
        self.task_type = task_type
        self.size = size
        self.max_length = max_length
        
        # Generate synthetic data based on task type
        if task_type == "listops":
            self.data = self._generate_listops_data()
        elif task_type == "text":
            self.data = self._generate_text_data()
        elif task_type == "retrieval":
            self.data = self._generate_retrieval_data()
        elif task_type == "image":
            self.data = self._generate_image_data()
        elif task_type == "pathfinder":
            self.data = self._generate_pathfinder_data()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _generate_listops_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic ListOps data.
        
        Returns:
            Tuple of (inputs, targets)
        """
        # Generate nested list operations with numbers and operators
        inputs = []
        targets = []
        
        for _ in range(self.size):
            # Random length sequence
            length = np.random.randint(10, min(200, self.max_length))
            
            # Generate sequence of numbers and operators
            sequence = []
            for _ in range(length):
                if np.random.random() < 0.7:  # 70% numbers
                    sequence.append(str(np.random.randint(0, 10)))
                else:  # 30% operators
                    sequence.append(np.random.choice(['[MAX', '[MIN', '[MED', ']']))
            
            # Convert to indices (simplified vocabulary)
            vocab = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                    '[MAX': 10, '[MIN': 11, '[MED': 12, ']': 13, '<PAD>': 14}
            
            input_ids = [vocab.get(token, 14) for token in sequence[:self.max_length]]
            # Pad to max_length
            input_ids.extend([14] * (self.max_length - len(input_ids)))
            
            inputs.append(input_ids)
            # Target: simplified result (0-9)
            targets.append(np.random.randint(0, 10))
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def _generate_text_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic text classification data.
        
        Returns:
            Tuple of (inputs, targets)
        """
        # Generate movie review-like text data
        inputs = []
        targets = []
        
        for _ in range(self.size):
            # Random length sequence
            length = np.random.randint(100, min(1000, self.max_length))
            
            # Generate word indices (simplified)
            input_ids = np.random.randint(0, 1000, length).tolist()
            # Pad to max_length
            input_ids.extend([0] * (self.max_length - len(input_ids)))
            
            inputs.append(input_ids)
            # Binary sentiment classification
            targets.append(np.random.randint(0, 2))
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def _generate_retrieval_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic document retrieval data.
        
        Returns:
            Tuple of (inputs, targets)
        """
        # Generate document retrieval data
        inputs = []
        targets = []
        
        for _ in range(self.size):
            # Two documents and a query
            doc1_length = np.random.randint(50, min(500, self.max_length // 3))
            doc2_length = np.random.randint(50, min(500, self.max_length // 3))
            query_length = np.random.randint(10, min(100, self.max_length // 3))
            
            # Generate document and query indices
            doc1 = np.random.randint(0, 1000, doc1_length).tolist()
            doc2 = np.random.randint(0, 1000, doc2_length).tolist()
            query = np.random.randint(0, 1000, query_length).tolist()
            
            # Combine with separators
            combined = doc1 + [1001] + doc2 + [1001] + query  # 1001 as separator
            combined = combined[:self.max_length]
            
            # Pad to max_length
            combined.extend([0] * (self.max_length - len(combined)))
            
            inputs.append(combined)
            # Binary matching task
            targets.append(np.random.randint(0, 2))
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def _generate_image_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic image classification data (pixel sequences).
        
        Returns:
            Tuple of (inputs, targets)
        """
        # Generate CIFAR-10 like pixel sequence data
        inputs = []
        targets = []
        
        for _ in range(self.size):
            # CIFAR-10: 32x32x3 = 3072 pixels
            # But we'll use a smaller sequence for synthetic data
            length = min(1024, self.max_length)
            
            # Generate pixel values (0-255)
            input_ids = np.random.randint(0, 256, length).tolist()
            # Pad to max_length
            input_ids.extend([0] * (self.max_length - len(input_ids)))
            
            inputs.append(input_ids)
            # 10-class classification
            targets.append(np.random.randint(0, 10))
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def _generate_pathfinder_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic pathfinder data.
        
        Returns:
            Tuple of (inputs, targets)
        """
        # Generate pathfinder-like data
        inputs = []
        targets = []
        
        for _ in range(self.size):
            # Grid-like data
            length = min(1024, self.max_length)
            
            # Generate grid values (0 for empty, 1 for path, 2 for start/end)
            input_ids = np.random.choice([0, 1, 2], length, p=[0.8, 0.15, 0.05]).tolist()
            # Pad to max_length
            input_ids.extend([0] * (self.max_length - len(input_ids)))
            
            inputs.append(input_ids)
            # Binary path existence
            targets.append(np.random.randint(0, 2))
        
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.data[0][idx], self.data[1][idx]


class LRABenchmark:
    """
    LRA Benchmark for Neural State Machine Models.
    """
    
    def __init__(self, model, device: torch.device = None):
        """
        Initialize LRA benchmark.
        
        Args:
            model: NSM model to benchmark
            device: Device to run benchmark on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def run_task(self, task_type: str, batch_size: int = 32, 
                 num_samples: int = 1000) -> Dict[str, float]:
        """
        Run LRA task benchmark.
        
        Args:
            task_type: LRA task type
            batch_size: Batch size for evaluation
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Running LRA {task_type} benchmark...")
        
        # Create dataset
        dataset = LRADataset(task_type=task_type, size=num_samples, max_length=1000)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs.float() if inputs.dtype == torch.long else inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
                
                # Limit for quick testing
                if batch_idx > 10:  # Just for demonstration
                    break
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else 0.0
        
        results = {
            'task': task_type,
            'accuracy': accuracy,
            'loss': avg_loss,
            'samples_processed': total
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Samples: {total}")
        
        return results
    
    def run_all_tasks(self, batch_size: int = 32, 
                     num_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Run all LRA tasks.
        
        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task
            
        Returns:
            Dictionary with results for all tasks
        """
        tasks = ["listops", "text", "retrieval", "image", "pathfinder"]
        results = {}
        
        for task in tasks:
            try:
                results[task] = self.run_task(task, batch_size, num_samples)
            except Exception as e:
                print(f"Error running {task}: {e}")
                results[task] = {
                    'task': task,
                    'error': str(e)
                }
        
        return results


# Example usage
if __name__ == "__main__":
    print("Testing LRA Benchmark...")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=1000, output_dim=10):
            super().__init__()
            self.embedding = nn.Embedding(1002, 128)  # +2 for padding and separators
            self.fc = nn.Linear(128, output_dim)
            
        def forward(self, x):
            # x shape: [batch_size, seq_len]
            embedded = self.embedding(x)  # [batch_size, seq_len, 128]
            # Global average pooling
            pooled = embedded.mean(dim=1)  # [batch_size, 128]
            output = self.fc(pooled)  # [batch_size, output_dim]
            return output
    
    # Test with simple model
    model = SimpleModel()
    benchmark = LRABenchmark(model)
    
    # Run a quick test
    results = benchmark.run_task("listops", batch_size=8, num_samples=100)
    print(f"Test results: {results}")
    
    print("✅ LRA Benchmark test completed!")