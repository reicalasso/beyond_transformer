# Literature Review: Beyond Transformers

This document summarizes the strengths, limitations, and alternatives to Transformer architectures, and positions the **Neural State Machine (NSM)** paradigm as a potential future cornerstone of AI.

---

## 1. Background

Transformers (Vaswani et al., 2017) revolutionized sequence modeling with self-attention. They became the backbone of state-of-the-art models in NLP (BERT, GPT, T5), vision (ViT), and multimodal architectures (CLIP, Flamingo).

---

## 2. Key Limitations of Transformers

1. **Quadratic Complexity**  
   - Attention requires O(n²) operations for sequence length *n*.  
   - Limits scalability to very long contexts.

2. **Sequence-Only Representation**  
   - Naturally optimized for linear token sequences.  
   - Less suitable for graph-structured, hierarchical, or multimodal data.

3. **Fixed Attention Mechanism**  
   - Attention weights are recomputed each layer, with no persistent memory.

4. **Compute & Memory Inefficiency**  
   - Requires large batch training and distributed compute.  
   - Inefficient for resource-constrained settings.

---

## 3. Alternative Directions

### Efficient Transformers
- **Linformer** (Wang et al., 2020): Low-rank projections reduce attention complexity.
- **Performer** (Choromanski et al., 2021): Random feature maps approximate softmax attention.
- **Reformer** (Kitaev et al., 2020): Locality-sensitive hashing for sparse attention.
- **Longformer** (Beltagy et al., 2020): Sliding window + global attention.

### State & Memory Models
- **RWKV** (Peng et al., 2023): RNN-like state, Transformer-style training.
- **S4** (Gu et al., 2022): Learnable ODE-based recurrence.
- **Retentive Networks** (Sun et al., 2023): Persistent state as an alternative to attention.

### Graph & Multimodal Extensions
- **Graph Attention Networks** (Veličković et al., 2018): Attention on graph structures.
- **Perceiver** (Jaegle et al., 2021): Latent bottleneck tokens for general perception across modalities.

---

## 4. Open Gaps

- **No Unified Memory-Attention Architecture**: There is no architecture that truly unifies persistent state with adaptive attention in a scalable way.
- **Fundamental Representation Limitations**: Most efficient Transformers only approximate attention costs but fail to address core issues with sequential representation.
- **Integration Gap**: The integration of graph reasoning with sequential memory is largely unexplored, despite its importance for real-world data.

---

## 5. Positioning Neural State Machines (NSM)

NSM is positioned to be a **game-changing paradigm** that addresses these critical gaps by:
- **Hybrid Architecture**: Combining **state-based memory (RNN-like)** with **adaptive attention (Transformer-like)** for the first time in a unified framework.
- **Efficient Computation**: Allowing tokens to connect only to important states, drastically reducing redundant computation and enabling **O(n·s)** complexity.
- **Universal Applicability**: Enabling efficient long-sequence modeling while being naturally applicable to graph, multimodal, and structured data.
- **Future of AI**: NSM has the potential to become the foundation for next-generation AI systems, offering better efficiency, scalability, and interpretability.

---

## 6. References

See `references/papers.bib` for full BibTeX entries.

- Vaswani et al., *Attention is All You Need*, NeurIPS 2017  
- Wang et al., *Linformer*, arXiv 2020  
- Choromanski et al., *Performer*, ICLR 2021  
- Kitaev et al., *Reformer*, ICLR 2020  
- Beltagy et al., *Longformer*, arXiv 2020  
- Gu et al., *S4: Structured State Space Models*, ICLR 2022  
- Peng et al., *RWKV: Reinventing RNNs for the Transformer Era*, 2023  
- Sun et al., *Retentive Networks*, NeurIPS 2023  
- Veličković et al., *Graph Attention Networks*, ICLR 2018  
- Jaegle et al., *Perceiver: General Perception with Iterative Attention*, ICML 2021  

---
*This document is a living resource and will be updated as new research emerges.*