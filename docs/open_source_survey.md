# Open Source Implementations Survey

This document catalogs relevant open-source implementations of Transformer alternatives and related architectures that could inform our research or provide comparison baselines.

## 1. GitHub Repositories

### 1.1 Mamba Implementations
- **state-spaces/mamba**: Official Mamba implementation
  - Location in our repo: `/workspaces/beyond_transformer/mamba`
  - Key features: Hardware-aware kernels, optimized for NVIDIA GPUs
  - Languages: Python, CUDA, C++

- **fla-org/flash-linear-attention**: Fast and memory-efficient linear attention
  - URL: https://github.com/fla-org/flash-linear-attention
  - Key features: Multiple efficient attention variants including Mamba

### 1.2 RWKV Implementations
- **BlinkDL/RWKV-LM**: Official RWKV implementation
  - Location in our repo: `/workspaces/beyond_transformer/RWKV-LM`
  - Key features: Multiple versions (v1-v7), various modalities
  - Languages: Python, PyTorch

### 1.3 Efficient Transformer Implementations
- **lucidrains/performer-pytorch**: Performer implementation
  - URL: https://github.com/lucidrains/performer-pytorch
  - Key features: Redesigned attention mechanism with linear complexity

- **lucidrains/reformer-pytorch**: Reformer implementation
  - URL: https://github.com/lucidrains/reformer-pytorch
  - Key features: Locality-sensitive hashing for efficient attention

- **lucidrains/linear-attention-transformer**: Linformer implementation
  - URL: https://github.com/lucidrains/linear-attention-transformer
  - Key features: Linear attention through low-rank projection

### 1.4 S4/Structured State Space Models
- **HazyResearch/state-spaces**: S4 implementation
  - Related to Mamba development
  - Key features: Structured state space models for sequence modeling

### 1.5 Graph Neural Networks
- **pyg-team/pytorch_geometric**: PyTorch Geometric for graph neural networks
  - URL: https://github.com/pyg-team/pytorch_geometric
  - Key features: Graph attention networks, various GNN implementations

## 2. Hugging Face Implementations

### 2.1 Transformers Library
- **huggingface/transformers**: Main Transformers library
  - URL: https://github.com/huggingface/transformers
  - Key features: Standard Transformers and some efficient variants

### 2.2 Mamba Integration
- **state-spaces/mamba**: Available through Hugging Face models
  - Integration with Transformers library for ease of use

### 2.3 RWKV Models
- **RWKV/**: Various RWKV models available on Hugging Face Hub
  - Pre-trained models for different tasks and sizes

## 3. Neural Turing Machine Implementations

### 3.1 Keras NTM
- Location in our repo: `/workspaces/beyond_transformer/ntm_keras`
- Key features: Educational implementation with detailed documentation

## 4. Specialized Libraries

### 4.1 FlashAttention
- **HazyResearch/flash-attention**: Fast and memory-efficient attention
  - URL: https://github.com/HazyResearch/flash-attention
  - Key features: Optimized attention kernels for NVIDIA GPUs

### 4.2 Triton Implementations
- Various repositories implementing efficient attention with Triton
  - Focus on custom GPU kernels for maximum efficiency

## 5. Benchmarking Suites

### 5.1 Long Range Arena (LRA)
- Standard benchmark for long-sequence models
- Includes tasks like ListOps, Text, Retrieval, Image, PathFinder

### 5.2 Other Benchmarks
- **GLUE/SuperGLUE**: Standard NLP benchmarks
- **ImageNet**: Standard computer vision benchmark
- **HELM**: Holistic evaluation of language models

## 6. Neural State Machine Implementations

### 6.1 Our Implementation
- Location: `/workspaces/beyond_transformer/src/nsm`
- Key features: Token-to-state routing, dynamic state management

### 6.2 Related Research Implementations
- Limited open-source implementations of similar concepts
- Most research in this area is still exploratory

## 7. Recommendations for Integration

1. **Mamba Integration**: Leverage existing Mamba implementation for baseline comparisons
2. **RWKV Reference**: Use RWKV as a representative RNN-style Transformer alternative
3. **Efficient Transformer Baselines**: Implement key efficient Transformer variants for comparison
4. **Graph Network Baselines**: Include GNN implementations for structured data tasks
5. **Standard Benchmarks**: Utilize established benchmarks for consistent evaluation

## 8. Future Scanning

We should regularly monitor:
- New releases from state-spaces (Mamba/S4)
- Updates to RWKV models
- Efficient attention implementations
- Hardware-optimized kernels
- New hybrid architectures

This survey will be updated as new relevant implementations are discovered or released.