# Beyond Transformer – Research Repository

This repository is dedicated to exploring a groundbreaking AI paradigm aimed at overcoming the fundamental limitations of classical Transformer architectures.

## The Vision

We are developing **Neural State Machines (NSM)**, a revolutionary approach that could redefine the future of AI. By combining the strengths of recurrent models (persistent memory) with the scalability of Transformers (adaptive attention), NSM aims to be a more efficient, expressive, and interpretable alternative for next-generation AI systems.

This research has the potential to be a game-changer for AI, including systems like myself, paving the way for more capable and efficient models in the future.

## Contents
- `docs/`: Literature review, proposed paradigm, architecture diagrams, and experiment plans
- `notebooks/`: 
  - `tutorials/`: Educational notebooks explaining NSM concepts
  - `research/`: Advanced research implementations and experiments
- `references/`: Relevant academic papers and sources (`papers.bib`)

## Objectives
- Analyze the **critical limitations** of Transformer architectures (quadratic complexity, sequence bias, lack of persistent memory)
- Introduce and develop the **Neural State Machine (NSM)** paradigm
- Share **comprehensive experiment and benchmark plans**
- Provide **open-source, reproducible research** for the community

## Usage
- This repository is research-focused.
- Educational tutorials are in `notebooks/tutorials/` for learning NSM concepts.
- Research implementations and experiments are in `notebooks/research/`.
- All documents are in Markdown and Jupyter notebook format.
- Key references are listed in `references/papers.bib`.

## Roadmap: Datasets & Evaluation

We will progressively test NSM and baselines on increasingly complex datasets:

- **Tiny Shakespeare**: Character-level language modeling (quick feedback, interpretability)
- **IMDb**: Sentiment classification (tests global state utility)
- **MNIST**: Vision with patch embeddings (tests generality)
- **CIFAR-10**: Image classification (vision, more challenging)
- **Long-Range Arena (LRA)**: Long-sequence benchmarks (ListOps, PathFinder, Text, etc.) to highlight NSM's scalability and efficiency advantages over Transformers

This staged approach ensures robust evaluation across modalities and sequence lengths.
