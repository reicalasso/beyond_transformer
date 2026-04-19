"""
PULSE Data Pipeline Utilities

Provides efficient data loading with:
- Sequence packing for better GPU utilization
- Dynamic batching
- Streaming support for large datasets
"""

import random
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, IterableDataset


class PackedDataset(Dataset):
    """
    Dataset that packs multiple sequences into fixed-length chunks.
    
    This improves training efficiency by:
    - Reducing padding waste
    - Better GPU utilization
    - More samples per batch
    
    Args:
        tokenized_texts: List of tokenized sequences
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID (used as separator)
    """
    
    def __init__(
        self,
        tokenized_texts: List[List[int]],
        max_length: int = 512,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Pack sequences
        self.packed_sequences = self._pack_sequences(tokenized_texts)
    
    def _pack_sequences(self, texts: List[List[int]]) -> List[List[int]]:
        """Pack multiple sequences into fixed-length chunks."""
        packed = []
        current_chunk = []
        
        for tokens in texts:
            # Add EOS separator between sequences
            if current_chunk:
                tokens_with_sep = [self.eos_token_id] + tokens
            else:
                tokens_with_sep = tokens
            
            # Check if adding this sequence would exceed max_length
            if len(current_chunk) + len(tokens_with_sep) <= self.max_length:
                current_chunk.extend(tokens_with_sep)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    packed.append(current_chunk)
                
                # Start new chunk
                # If sequence is longer than max_length, split it
                if len(tokens) > self.max_length:
                    for i in range(0, len(tokens), self.max_length):
                        chunk = tokens[i:i + self.max_length]
                        if len(chunk) > self.max_length // 4:  # Only keep substantial chunks
                            packed.append(chunk)
                    current_chunk = []
                else:
                    current_chunk = tokens
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) > self.max_length // 4:
            packed.append(current_chunk)
        
        return packed
    
    def __len__(self) -> int:
        return len(self.packed_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.packed_sequences[idx]
        
        # Pad if needed
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [self.pad_token_id] * (self.max_length + 1 - len(tokens))
        else:
            tokens = tokens[:self.max_length + 1]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Input and target (shifted by 1)
        return tokens[:-1], tokens[1:]


class StreamingPackedDataset(IterableDataset):
    """
    Streaming dataset with packing for large datasets.
    
    Processes data on-the-fly without loading everything into memory.
    
    Args:
        data_iterator: Iterator yielding tokenized sequences
        max_length: Maximum sequence length
        buffer_size: Number of sequences to buffer for packing
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID
    """
    
    def __init__(
        self,
        data_iterator: Iterator[List[int]],
        max_length: int = 512,
        buffer_size: int = 1000,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
        shuffle_buffer: bool = True,
    ):
        self.data_iterator = data_iterator
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        buffer = []
        current_chunk = []
        
        for tokens in self.data_iterator:
            buffer.append(tokens)
            
            # Process buffer when full
            if len(buffer) >= self.buffer_size:
                if self.shuffle_buffer:
                    random.shuffle(buffer)
                
                for seq in buffer:
                    # Pack sequences
                    if current_chunk:
                        seq_with_sep = [self.eos_token_id] + seq
                    else:
                        seq_with_sep = seq
                    
                    if len(current_chunk) + len(seq_with_sep) <= self.max_length:
                        current_chunk.extend(seq_with_sep)
                    else:
                        if current_chunk:
                            yield self._prepare_sample(current_chunk)
                        
                        if len(seq) > self.max_length:
                            for i in range(0, len(seq), self.max_length):
                                chunk = seq[i:i + self.max_length]
                                if len(chunk) > self.max_length // 4:
                                    yield self._prepare_sample(chunk)
                            current_chunk = []
                        else:
                            current_chunk = seq
                
                buffer = []
        
        # Process remaining buffer
        for seq in buffer:
            if current_chunk:
                seq_with_sep = [self.eos_token_id] + seq
            else:
                seq_with_sep = seq
            
            if len(current_chunk) + len(seq_with_sep) <= self.max_length:
                current_chunk.extend(seq_with_sep)
            else:
                if current_chunk:
                    yield self._prepare_sample(current_chunk)
                current_chunk = seq if len(seq) <= self.max_length else []
        
        if current_chunk:
            yield self._prepare_sample(current_chunk)
    
    def _prepare_sample(self, tokens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare a single sample with padding."""
        if len(tokens) < self.max_length + 1:
            tokens = tokens + [self.pad_token_id] * (self.max_length + 1 - len(tokens))
        else:
            tokens = tokens[:self.max_length + 1]
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens[:-1], tokens[1:]


class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups sequences by length.
    
    This reduces padding waste by batching similar-length sequences together.
    
    Args:
        lengths: List of sequence lengths
        max_tokens: Maximum tokens per batch
        max_sentences: Maximum sentences per batch
        shuffle: Whether to shuffle batches
    """
    
    def __init__(
        self,
        lengths: List[int],
        max_tokens: int = 4096,
        max_sentences: int = 32,
        shuffle: bool = True,
    ):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.shuffle = shuffle
        
        # Create batches
        self.batches = self._create_batches()
    
    def _create_batches(self) -> List[List[int]]:
        """Create batches grouped by length."""
        # Sort indices by length
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i])
        
        batches = []
        current_batch = []
        current_tokens = 0
        current_max_len = 0
        
        for idx in indices:
            seq_len = self.lengths[idx]
            
            # Check if adding this sequence would exceed limits
            new_max_len = max(current_max_len, seq_len)
            new_tokens = new_max_len * (len(current_batch) + 1)
            
            if (new_tokens > self.max_tokens or 
                len(current_batch) >= self.max_sentences):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = seq_len
                current_tokens = seq_len
            else:
                current_batch.append(idx)
                current_max_len = new_max_len
                current_tokens = new_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            random.shuffle(self.batches)
        
        for batch in self.batches:
            if self.shuffle:
                random.shuffle(batch)
            yield batch
    
    def __len__(self) -> int:
        return len(self.batches)


def collate_with_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that pads sequences to the same length.
    
    Args:
        batch: List of (input, target) tuples
        pad_token_id: Token ID to use for padding
        
    Returns:
        Padded input and target tensors
    """
    inputs, targets = zip(*batch)
    
    # Find max length in batch
    max_len = max(x.shape[0] for x in inputs)
    
    # Pad sequences
    padded_inputs = torch.full((len(inputs), max_len), pad_token_id, dtype=torch.long)
    padded_targets = torch.full((len(targets), max_len), -100, dtype=torch.long)  # -100 for ignore
    
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :inp.shape[0]] = inp
        padded_targets[i, :tgt.shape[0]] = tgt
    
    return padded_inputs, padded_targets


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Create attention mask from input IDs.
    
    Args:
        input_ids: Input token IDs [batch, seq_len]
        pad_token_id: Padding token ID
        
    Returns:
        Attention mask [batch, seq_len] (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def estimate_dataset_tokens(
    dataset: Dataset,
    sample_size: int = 1000,
) -> int:
    """
    Estimate total tokens in dataset by sampling.
    
    Args:
        dataset: Dataset to estimate
        sample_size: Number of samples to use for estimation
        
    Returns:
        Estimated total tokens
    """
    sample_size = min(sample_size, len(dataset))
    indices = random.sample(range(len(dataset)), sample_size)
    
    total_tokens = 0
    for idx in indices:
        x, _ = dataset[idx]
        total_tokens += (x != 0).sum().item()  # Assuming 0 is pad
    
    avg_tokens = total_tokens / sample_size
    return int(avg_tokens * len(dataset))
