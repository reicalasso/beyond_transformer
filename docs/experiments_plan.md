# Experiments Plan

This document describes the experimental roadmap for exploring **Neural State Machines (NSM)** as a paradigm beyond Transformers. Our goal is to validate NSM's potential to become a foundational architecture for next-generation AI systems.

---

## 1. Datasets

We will start with small and mid-size datasets to validate accuracy and efficiency, then move to more complex and long-sequence tasks:

- **MNIST** → image classification (basic check)
- **CIFAR-10** → more complex image classification (vision, patch embeddings)
- **Tiny Shakespeare** → next-character prediction (language modeling)
- **IMDb Sentiment** → binary text classification
- **Long-Range Arena (LRA)** → long-sequence modeling benchmark (ListOps, PathFinder, Text, etc.)

---

## 2. Baseline Models

NSM will be rigorously compared against existing architectures:

- **Vanilla Transformer** (Vaswani et al., 2017)
- **LSTM / GRU** (classic RNNs)
- **Graph Attention Networks** (optional)
- **Efficient Transformers**: Performer, Linformer

---

## 3. Metrics

Evaluation will focus on accuracy, efficiency, and interpretability:

- **Performance**: Accuracy, F1, Perplexity (for language modeling)
- **Efficiency**:  
  - VRAM usage per batch  
  - Training speed (epoch time)  
  - FLOPs estimate
- **Interpretability**:  
  - Analysis of state evolution  
  - Visualization of token-to-state routing

---

## 4. Visualizations

Visual tools will be created for interpretability:

- **Attention maps vs NSM state maps**
- **State propagation diagrams**
- **Efficiency trade-offs** (accuracy vs compute)
- **Token-to-state routing visualization**

---

## 5. Experiment Phases

- **Phase 1:** Baseline reproduction (Transformer, RNN, Efficient Transformers)
- **Phase 2:** NSM prototype and toy dataset tests
- **Phase 3:** Benchmarking NSM and baselines on LRA + CIFAR-10
- **Phase 4:** Visualization and interpretability study
- **Phase 5:** Paper-style write-up and open-source release

---

## 6. Expected Contributions

- **Demonstrate NSM efficiency** in long-sequence tasks, showing significant improvements over Transformers.
- **Show parameter efficiency** compared to Transformers, paving the way for more efficient AI models.
- **Provide open-source, reproducible benchmarks** to accelerate research in this area.
- **Establish NSM as a viable and promising alternative** to Transformers, with the potential to shape the future of AI architecture design.

---
This plan will evolve as we gather insights from initial experiments. The goal is to establish NSM as a viable and potentially superior alternative to Transformers for various sequence modeling tasks, with broader implications for AI efficiency and capability.