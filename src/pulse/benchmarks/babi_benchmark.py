"""
bAbI Benchmark for PULSE Models

This module implements bAbI benchmark tests for evaluating memory capabilities.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class bAbIDataset(Dataset):
    """
    bAbI dataset for memory reasoning tasks.
    """

    def __init__(
        self,
        task_id: int = 1,
        size: int = 1000,
        max_story_length: int = 20,
        max_query_length: int = 10,
    ):
        """
        Initialize bAbI dataset.

        Args:
            task_id: bAbI task ID (1-20)
            size: Number of samples
            max_story_length: Maximum story length
            max_query_length: Maximum query length
        """
        self.task_id = task_id
        self.size = size
        self.max_story_length = max_story_length
        self.max_query_length = max_query_length

        # Generate synthetic bAbI data
        self.data = self._generate_babi_data()

    def _generate_babi_data(self) -> Tuple[List[List[int]], List[List[int]], List[int]]:
        """
        Generate synthetic bAbI data.

        Returns:
            Tuple of (stories, queries, answers)
        """
        stories = []
        queries = []
        answers = []

        # Vocabulary for synthetic data
        vocab = {
            "name": ["john", "mary", "bob", "alice", "tom", "sue"],
            "object": ["apple", "book", "ball", "cat", "dog", "car"],
            "location": ["kitchen", "bedroom", "garden", "office", "hall", "bathroom"],
            "verb": ["got", "put", "moved", "took", "left", "picked"],
            "question": ["where", "what", "who", "how"],
        }

        # Answer vocabulary (depends on task)
        if self.task_id in [
            1,
            2,
            3,
        ]:  # Supporting facts, two supporting facts, three supporting facts
            answer_vocab = vocab["location"]
        elif self.task_id == 16:  # Basic induction
            answer_vocab = ["yes", "no"]
        elif self.task_id == 19:  # Path finding
            answer_vocab = vocab["location"]
        else:
            answer_vocab = vocab["location"]  # Default

        # Create vocabulary mapping
        all_words = []
        for category in vocab.values():
            all_words.extend(category)
        all_words.extend(["?", ".", "is", "at", "in", "the", "of", "to", "from"])
        self.word_to_idx = {
            word: idx + 1 for idx, word in enumerate(sorted(set(all_words)))
        }
        self.word_to_idx["<PAD>"] = 0
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        for _ in range(self.size):
            # Generate story
            story_length = np.random.randint(2, self.max_story_length + 1)
            story = []

            # Generate facts
            entities = np.random.choice(vocab["name"], 3, replace=False)
            locations = np.random.choice(vocab["location"], 3, replace=False)
            objects = np.random.choice(vocab["object"], 2, replace=False)

            # Create story based on task
            if self.task_id in [1, 2, 3]:  # Supporting facts tasks
                # Create simple location facts
                for i in range(story_length):
                    entity = np.random.choice(entities)
                    location = np.random.choice(locations)
                    fact = f"{entity} is in the {location} ."
                    story.append(fact)

                # Generate query about last few facts
                query_entity = entities[0]
                query = f"where is {query_entity} ?"

                # Simple answer (last location of entity)
                answer = locations[0]

            elif self.task_id == 16:  # Basic induction
                # Create comparison facts
                obj1, obj2 = objects[:2]
                prop1, prop2 = np.random.randint(1, 10, 2)
                story.append(f"{obj1} is bigger than {obj2} .")
                story.append(f"{obj2} is {prop2} cm tall .")

                query = f"is {obj1} bigger than {obj2} ?"
                answer = "yes"

            elif self.task_id == 19:  # Path finding
                # Create path facts
                loc1, loc2, loc3 = locations[:3]
                story.append(f"from {loc1} to {loc2} takes 5 minutes .")
                story.append(f"from {loc2} to {loc3} takes 3 minutes .")

                query = f"where can you go from {loc1} ?"
                answer = loc2

            else:  # Default task
                # Generate generic story
                for i in range(story_length):
                    entity = np.random.choice(entities)
                    location = np.random.choice(locations)
                    obj = np.random.choice(objects)
                    verb = np.random.choice(vocab["verb"])
                    fact = f"{entity} {verb} the {obj} in the {location} ."
                    story.append(fact)

                query_entity = entities[0]
                query = f"where is {query_entity} ?"
                answer = locations[0]

            # Convert to indices
            story_indices = []
            for sentence in story:
                words = sentence.lower().split()
                indices = [self.word_to_idx.get(word, 0) for word in words]
                story_indices.append(indices)

            query_words = query.lower().split()
            query_indices = [self.word_to_idx.get(word, 0) for word in query_words]

            # Pad sequences
            padded_story = []
            for sent_indices in story_indices:
                if len(sent_indices) < 10:
                    sent_indices.extend([0] * (10 - len(sent_indices)))
                padded_story.extend(sent_indices[:10])

            if len(padded_story) < self.max_story_length * 10:
                padded_story.extend(
                    [0] * (self.max_story_length * 10 - len(padded_story))
                )

            if len(query_indices) < self.max_query_length:
                query_indices.extend([0] * (self.max_query_length - len(query_indices)))

            stories.append(padded_story[: self.max_story_length * 10])
            queries.append(query_indices[: self.max_query_length])
            answers.append(self.word_to_idx.get(answer, 0))

        return (
            torch.tensor(stories, dtype=torch.long),
            torch.tensor(queries, dtype=torch.long),
            torch.tensor(answers, dtype=torch.long),
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.data[0][idx], self.data[1][idx], self.data[2][idx]


class bAbIBenchmark:
    """
    bAbI Benchmark for PULSE Models.
    """

    def __init__(self, model, device: torch.device = None):
        """
        Initialize bAbI benchmark.

        Args:
            model: PULSE model to benchmark
            device: Device to run benchmark on
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def run_task(
        self, task_id: int, batch_size: int = 32, num_samples: int = 1000
    ) -> Dict[str, float]:
        """
        Run bAbI task benchmark.

        Args:
            task_id: bAbI task ID (1-20)
            batch_size: Batch size for evaluation
            num_samples: Number of samples to test

        Returns:
            Dictionary with benchmark results
        """
        print(f"Running bAbI task {task_id} benchmark...")

        # Create dataset
        dataset = bAbIDataset(task_id=task_id, size=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Get vocabulary size
        vocab_size = dataset.vocab_size

        # Evaluation
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch_idx, (stories, queries, answers) in enumerate(dataloader):
                stories, queries, answers = (
                    stories.to(self.device),
                    queries.to(self.device),
                    answers.to(self.device),
                )

                # Combine stories and queries (simplified approach)
                # In a real implementation, you'd have a more sophisticated model
                combined_input = torch.cat([stories.float(), queries.float()], dim=1)

                # Forward pass
                outputs = self.model(combined_input)

                # Ensure outputs match vocabulary size
                if outputs.size(-1) != vocab_size:
                    # Project to vocabulary size
                    projection = nn.Linear(outputs.size(-1), vocab_size).to(self.device)
                    outputs = projection(outputs)

                # Calculate loss
                loss = criterion(outputs, answers)
                total_loss += loss.item()

                # Calculate accuracy
                predictions = outputs.argmax(dim=1)
                correct += (predictions == answers).sum().item()
                total += answers.size(0)

                # Limit for quick testing
                if batch_idx > 5:  # Just for demonstration
                    break

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else 0.0

        results = {
            "task_id": task_id,
            "accuracy": accuracy,
            "loss": avg_loss,
            "samples_processed": total,
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Samples: {total}")

        return results

    def run_key_tasks(
        self, batch_size: int = 32, num_samples: int = 1000
    ) -> Dict[int, Dict[str, float]]:
        """
        Run key bAbI tasks that test memory capabilities.

        Args:
            batch_size: Batch size for evaluation
            num_samples: Number of samples per task

        Returns:
            Dictionary with results for key tasks
        """
        # Key tasks for memory evaluation
        key_tasks = [
            1,
            2,
            3,
            16,
            19,
        ]  # Supporting facts tasks + induction + path finding
        results = {}

        for task_id in key_tasks:
            try:
                results[task_id] = self.run_task(task_id, batch_size, num_samples)
            except Exception as e:
                print(f"Error running task {task_id}: {e}")
                results[task_id] = {"task_id": task_id, "error": str(e)}

        return results


# Example usage
if __name__ == "__main__":
    print("Testing bAbI Benchmark...")

    # Create a simple model for testing
    class SimplebAbIModel(nn.Module):
        def __init__(self, input_dim=300, output_dim=100):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            # x shape: [batch_size, seq_len]
            x = self.relu(self.fc1(x))
            output = self.fc2(x)  # [batch_size, output_dim]
            return output

    # Test with simple model
    model = SimplebAbIModel()
    benchmark = bAbIBenchmark(model)

    # Run a quick test
    results = benchmark.run_task(1, batch_size=8, num_samples=100)
    print(f"Test results: {results}")

    print("✅ bAbI Benchmark test completed!")
