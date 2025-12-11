# Experiments Plan


This document describes the experimental roadmap for exploring **PULSEs (PULSE)** as a paradigm beyond Transformers. The project yapısı ve kodlar için güncel referanslar aşağıdadır:

## Project Directory Structure (2025)

```
requirements/                # Gereksinim dosyaları
  requirements.txt
  requirements-experiments.txt
  requirements-test.txt
results/
  experiments/             # Deney sonuçları (json)
  visualization/           # Görselleştirme çıktıları (png)
scripts/
  experiments/             # Deney çalıştırma ve özetleme scriptleri
  visualization/           # Görselleştirme scriptleri
src/                         # Ana kaynak kodu
docs/                        # Belgeler
notebooks/
  experiments/             # Ana deney defterleri (ipynb)
  scripts/                 # Deney ve görselleştirme scriptleri (py)
  tests/                   # Test notebook ve scriptleri
  utils/                   # Yardımcı scriptler
  research/                # Araştırma amaçlı notebooklar
  tutorials/               # Eğitim ve örnek notebooklar
references/                  # Referanslar (papers.bib)
tests/                       # Testler
```

Referanslar için: `references/papers.bib`

Kod ve notebook örnekleri için:
- Ana deney: `notebooks/experiments/baseline_comparison.ipynb`
- Hiperparametre süpürme: `notebooks/experiments/hyperparameter_sweep.ipynb`
- Routing görselleştirme: `notebooks/experiments/routing_viz.ipynb`
- Script örnekleri: `scripts/experiments/run_experiments.py`, `scripts/visualization/plot_results.py`

Sonuç dosyaları: `results/experiments/`, görseller: `results/visualization/`

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

PULSE will be rigorously compared against existing architectures:

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

- **Attention maps vs PULSE state maps**
- **State propagation diagrams**
- **Efficiency trade-offs** (accuracy vs compute)
- **Token-to-state routing visualization**

---

## 5. Experiment Phases

- **Phase 1:** Baseline reproduction (Transformer, RNN, Efficient Transformers)
- **Phase 2:** PULSE prototype and toy dataset tests
- **Phase 3:** Benchmarking PULSE and baselines on LRA + CIFAR-10
- **Phase 4:** Visualization and interpretability study
- **Phase 5:** Paper-style write-up and open-source release

---

## 6. Expected Contributions

- **Demonstrate PULSE efficiency** in long-sequence tasks, showing significant improvements over Transformers.
- **Show parameter efficiency** compared to Transformers, paving the way for more efficient AI models.
- **Provide open-source, reproducible benchmarks** to accelerate research in this area.
- **Establish PULSE as a viable and promising alternative** to Transformers, with the potential to shape the future of AI architecture design.

---
This plan will evolve as we gather insights from initial experiments. The goal is to establish PULSE as a viable and potentially superior alternative to Transformers for various sequence modeling tasks, with broader implications for AI efficiency and capability.