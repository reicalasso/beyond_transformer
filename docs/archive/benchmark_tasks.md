# Benchmark Tasks for PULSE Evaluation

This document identifies benchmark tasks that require long-term memory and are suitable for evaluating PULSEs.

## 1. Long-term Memory Requiring Tasks

### 1.1 Long Range Arena (LRA)
**Description**: A benchmark suite designed to evaluate the ability of models to handle long sequences.
**Components**:
- ListOps: Nested list-based operations requiring hierarchical reasoning
- Text: Document classification on IMDb movie reviews
- Retrieval: Document matching task on ArXiv papers abstracts
- Image: Pixel-level sequential classification on grayscale CIFAR-10
- Pathfinder: Path detection in images with varying path lengths
- PathX: More complex path detection with cluttered backgrounds

**Why it's suitable for PULSE**:
- Tests long-sequence processing capabilities
- Requires memory retention across long contexts
- Diverse task types evaluate different aspects of memory

### 1.2 bAbI Tasks
**Description**: A set of 20 synthetic question-answering tasks designed to test text understanding and reasoning.
**Key Tasks for Memory Evaluation**:
- Task 1: 1 supporting fact
- Task 2: 2 supporting facts
- Task 3: 3 supporting facts
- Task 16: Basic induction
- Task 18: Reasoning about size
- Task 19: Path finding

**Why it's suitable for PULSE**:
- Explicitly tests memory and reasoning capabilities
- Requires tracking information across multiple sentences
- Well-defined evaluation metrics

### 1.3 PG-19 (Project Gutenberg 19)
**Description**: A large dataset of books from Project Gutenberg for language modeling.
**Characteristics**:
- 18GB of text from 28,000 books
- Sequences up to 1 million tokens
- Language modeling benchmark for long-context understanding

**Why it's suitable for PULSE**:
- Extremely long sequences test memory scalability
- Real-world text complexity
- Evaluates long-term dependency modeling

### 1.4 MemN2N (Memory Networks)
**Description**: Synthetic question answering tasks specifically designed to test memory capabilities.
**Task Types**:
- Single supporting fact QA
- Two supporting facts QA
- Three supporting facts QA
- Conjunction QA
- Time reasoning QA

**Why it's suitable for PULSE**:
- Explicitly designed to test memory mechanisms
- Controlled difficulty levels
- Direct comparison with memory-based architectures

## 2. Additional Relevant Benchmarks

### 2.1 SCROLLS (Standardized CompaRison Over Long Sequences)
**Components**:
- Quality: Question answering on narrative documents
- NarrativeQA: Summary-level question answering
- Qasper: Question answering on NLP papers
- QuALITY: Multiple choice questions on literary works
- ContractNLI: Natural language inference on legal contracts
- GovReport: Summarization of government reports

**Why it's suitable for PULSE**:
- Real-world long document understanding
- Diverse task formats
- Standardized evaluation

### 2.2 BigBird Benchmarks
Tasks used in the BigBird paper:
- PubMed abstracts classification
- IMDB movie reviews
- AG News classification
- Long-range language modeling

**Why it's suitable for PULSE**:
- Established benchmark for long-sequence models
- Document-level understanding tasks

### 2.3 Variable Computation Tasks
Tasks that require adaptive computation:
- Adaptive computation time benchmarks
- Neural Turing Machine tasks
- Differentiable neural computer tasks

**Why it's suitable for PULSE**:
- Tests dynamic state management
- Evaluates computational efficiency

## 3. Task Selection Rationale

### 3.1 Primary Benchmark Suite: LRA
**Rationale**:
- Standardized benchmark for long-sequence models
- Diverse task types (classification, matching, pixel-level)
- Well-established evaluation protocol
- Used by many recent long-sequence architectures

### 3.2 Secondary Benchmarks
1. **bAbI Tasks**: For explicit memory reasoning evaluation
2. **SCROLLS**: For real-world document understanding
3. **MemN2N**: For direct comparison with memory architectures

## 4. Task Characteristics Summary

| Benchmark | Sequence Length | Memory Requirements | Task Type | Evaluation Metric |
|-----------|----------------|---------------------|-----------|-------------------|
| LRA ListOps | Up to 2K tokens | High | Hierarchical reasoning | Accuracy |
| LRA Text | Up to 4K tokens | Medium | Document classification | Accuracy |
| LRA Retrieval | Up to 4K tokens | High | Document matching | Accuracy |
| LRA Image | Up to 1K pixels | Medium | Pixel classification | Accuracy |
| LRA Pathfinder | Up to 1K pixels | Medium | Path detection | Accuracy |
| bAbI | Variable | High | QA with reasoning | Accuracy |
| PG-19 | Up to 1M tokens | Very High | Language modeling | Perplexity |
| MemN2N | Variable | High | Memory QA | Accuracy |
| SCROLLS | Up to 80K tokens | High | Document QA/Summarization | Task-specific |

## 5. Implementation Considerations

### 5.1 Dataset Availability
- LRA: Publicly available through TensorFlow Datasets
- bAbI: Available through various sources including Facebook AI
- PG-19: Available through official sources
- MemN2N: Synthetic tasks that can be generated
- SCROLLS: Available through Hugging Face datasets

### 5.2 Evaluation Infrastructure
- Standard metrics (accuracy, perplexity, F1)
- Memory profiling tools
- Timing measurements
- Visualization capabilities for interpretability

## 6. Recommended Task Selection

### 6.1 Phase 1: Core Evaluation
1. **LRA Suite** (all tasks):
   - ListOps
   - Text (IMDb)
   - Retrieval (ArXiv)
   - Image (CIFAR-10)
   - Pathfinder
   - PathX

2. **bAbI Tasks** (select key tasks):
   - Task 1, 2, 3 (increasing memory requirements)
   - Task 16 (induction)
   - Task 19 (path finding)

### 6.2 Phase 2: Extended Evaluation
1. **SCROLLS Suite** (select representative tasks):
   - NarrativeQA or Qasper
   - ContractNLI
   - GovReport

2. **MemN2N Tasks**:
   - Synthetic memory tasks for controlled evaluation

### 6.3 Phase 3: Extreme Long-range Evaluation
1. **PG-19 Subset**:
   - Language modeling on long sequences
   - Evaluation of memory scalability

## 7. Baseline Models for Comparison

To establish meaningful benchmarks, we should compare against:

1. **Traditional Transformers** (where feasible)
2. **Efficient Transformers** (Longformer, BigBird, etc.)
3. **RNN-based Models** (LSTM, GRU, RWKV)
4. **State Space Models** (S4, Mamba)
5. **Memory-Augmented Models** (NTM, DNC)

## 8. Conclusion

The selected benchmarks provide a comprehensive evaluation of long-term memory capabilities across:
- Synthetic tasks with controlled difficulty
- Real-world document understanding
- Extreme long-sequence processing
- Diverse task formats (classification, QA, generation)

This benchmark suite will enable us to thoroughly evaluate the PULSE's capabilities in comparison to existing approaches.