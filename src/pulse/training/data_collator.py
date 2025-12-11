"""
Data Collators for PULSE Training.

This module provides data collation utilities for different tasks:
- DataCollatorForLanguageModeling: For causal and masked LM
- DataCollatorForSequenceClassification: For classification tasks
- DataCollatorForTokenClassification: For token-level tasks
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling tasks.
    
    Handles padding, attention masks, and label creation for
    both causal LM and masked LM tasks.
    """

    pad_token_id: int = 0
    mlm: bool = False
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    padding: str = "longest"  # "longest", "max_length", "do_not_pad"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries with 'input_ids' key.

        Returns:
            Batched tensors with padding and attention masks.
        """
        # Extract input_ids
        input_ids_list = [f["input_ids"] for f in features]

        # Convert to tensors if needed
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]

        # Determine max length
        if self.padding == "longest":
            max_len = max(ids.size(0) for ids in input_ids_list)
        elif self.padding == "max_length" and self.max_length is not None:
            max_len = self.max_length
        else:
            # No padding - all sequences must be same length
            max_len = input_ids_list[0].size(0)

        # Pad sequences
        batch_size = len(input_ids_list)
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            length = min(ids.size(0), max_len)
            padded_input_ids[i, :length] = ids[:length]
            attention_mask[i, :length] = 1

        batch = {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
        }

        # Create labels for language modeling
        if self.mlm:
            # Masked language modeling
            labels, masked_input_ids = self._mask_tokens(padded_input_ids.clone())
            batch["input_ids"] = masked_input_ids
            batch["labels"] = labels
        else:
            # Causal language modeling - labels are same as input_ids
            labels = padded_input_ids.clone()
            # Set padding tokens to -100 (ignored in loss)
            labels[labels == self.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def _mask_tokens(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mask tokens for masked language modeling.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Tuple of (labels, masked_input_ids)
        """
        labels = input_ids.clone()

        # Create probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask padding tokens
        padding_mask = input_ids == self.pad_token_id
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set non-masked tokens to -100 (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace with [MASK] token (assume token id 103)
        mask_token_id = 103
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = mask_token_id

        # 10% of the time, replace with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(1000, labels.shape, dtype=torch.long)  # Assume vocab size > 1000
        input_ids[indices_random] = random_words[indices_random]

        # 10% of the time, keep original token (do nothing)

        return labels, input_ids


@dataclass
class DataCollatorForSequenceClassification:
    """
    Data collator for sequence classification tasks.
    """

    pad_token_id: int = 0
    max_length: Optional[int] = None
    padding: str = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries with 'input_ids' and 'label' keys.

        Returns:
            Batched tensors with padding and attention masks.
        """
        # Extract input_ids and labels
        input_ids_list = [f["input_ids"] for f in features]
        labels = [f["label"] for f in features]

        # Convert to tensors if needed
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]

        # Determine max length
        if self.padding == "longest":
            max_len = max(ids.size(0) for ids in input_ids_list)
        elif self.padding == "max_length" and self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = input_ids_list[0].size(0)

        # Pad sequences
        batch_size = len(input_ids_list)
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, ids in enumerate(input_ids_list):
            length = min(ids.size(0), max_len)
            padded_input_ids[i, :length] = ids[:length]
            attention_mask[i, :length] = 1

        # Convert labels to tensor
        if isinstance(labels[0], (int, float)):
            labels_tensor = torch.tensor(labels, dtype=torch.long)
        else:
            labels_tensor = torch.stack(labels)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }


@dataclass
class DataCollatorForTokenClassification:
    """
    Data collator for token classification tasks (NER, POS tagging).
    """

    pad_token_id: int = 0
    label_pad_token_id: int = -100
    max_length: Optional[int] = None
    padding: str = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries with 'input_ids' and 'labels' keys.

        Returns:
            Batched tensors with padding and attention masks.
        """
        # Extract input_ids and labels
        input_ids_list = [f["input_ids"] for f in features]
        labels_list = [f["labels"] for f in features]

        # Convert to tensors if needed
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
        if not isinstance(labels_list[0], torch.Tensor):
            labels_list = [torch.tensor(labels, dtype=torch.long) for labels in labels_list]

        # Determine max length
        if self.padding == "longest":
            max_len = max(ids.size(0) for ids in input_ids_list)
        elif self.padding == "max_length" and self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = input_ids_list[0].size(0)

        # Pad sequences
        batch_size = len(input_ids_list)
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), self.label_pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, (ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            length = min(ids.size(0), max_len)
            padded_input_ids[i, :length] = ids[:length]
            padded_labels[i, :length] = labels[:length]
            attention_mask[i, :length] = 1

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }


@dataclass
class DataCollatorWithPadding:
    """
    Generic data collator with padding support.
    """

    pad_token_id: int = 0
    max_length: Optional[int] = None
    padding: str = "longest"
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries.

        Returns:
            Batched tensors with padding.
        """
        # Get all keys from first feature
        keys = features[0].keys()

        batch = {}
        for key in keys:
            values = [f[key] for f in features]

            # Handle different types
            if isinstance(values[0], torch.Tensor):
                if values[0].dim() == 0:
                    # Scalar tensors
                    batch[key] = torch.stack(values)
                else:
                    # Sequence tensors - need padding
                    batch[key] = self._pad_sequence(values, key)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            elif isinstance(values[0], list):
                # Convert lists to tensors and pad
                tensor_values = [torch.tensor(v) for v in values]
                batch[key] = self._pad_sequence(tensor_values, key)
            else:
                # Keep as-is for other types
                batch[key] = values

        return batch

    def _pad_sequence(
        self, sequences: List[torch.Tensor], key: str
    ) -> torch.Tensor:
        """Pad a list of sequences to the same length."""
        # Determine max length
        if self.padding == "longest":
            max_len = max(seq.size(0) for seq in sequences)
        elif self.padding == "max_length" and self.max_length is not None:
            max_len = self.max_length
        else:
            max_len = sequences[0].size(0)

        # Determine padding value
        if "label" in key.lower():
            pad_value = -100
        else:
            pad_value = self.pad_token_id

        # Pad sequences
        batch_size = len(sequences)
        padded = torch.full((batch_size, max_len), pad_value, dtype=sequences[0].dtype)

        for i, seq in enumerate(sequences):
            length = min(seq.size(0), max_len)
            padded[i, :length] = seq[:length]

        return padded
