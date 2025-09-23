Architecture Design
==================

Neural State Machines represent a paradigm shift from attention-based models to state-based computation, achieving linear complexity while maintaining competitive performance.

Core Principles
---------------

State-Based Computation
~~~~~~~~~~~~~~~~~~~~~~~

Unlike Transformers that compute attention over all token pairs (O(n²) complexity), NSMs maintain a fixed number of state vectors that evolve over time:

.. math::

   \text{Complexity}_{\text{Transformer}} = O(n^2 \cdot d)
   
   \text{Complexity}_{\text{NSM}} = O(s \cdot d)

where :math:`n` is sequence length, :math:`s` is number of states, and :math:`d` is model dimension.

Information Flow
~~~~~~~~~~~~~~~~

The NSM architecture processes information through four key stages:

1. **Token-to-State Routing**: Maps input tokens to state vectors
2. **State Propagation**: Updates states using gated mechanisms  
3. **State Communication**: Enables information flow between states
4. **State-to-Token Projection**: Generates output predictions

.. code-block:: text

   Input Tokens → [Routing] → State Vectors → [Propagation] → Updated States
                                    ↓              ↑
                              [Communication] ←    →
                                    ↓
   Output Logits ← [Projection] ← Final States

Key Components
--------------

Token-to-State Router
~~~~~~~~~~~~~~~~~~~~~

The router determines how input information is distributed across states:

.. math::

   \text{routing\_weights} = \text{softmax}(W_r \cdot \text{input\_embeddings})
   
   \text{state\_updates} = \sum_{t} \text{routing\_weights}_{t,s} \cdot \text{input}_{t}

**Key Properties:**

* **Sparsity**: Each token typically routes to a subset of states
* **Adaptivity**: Routing patterns learned during training
* **Efficiency**: O(n·s) complexity for routing computation

State Propagator
~~~~~~~~~~~~~~~~

The propagator updates state vectors using gated mechanisms similar to LSTMs/GRUs:

.. math::

   \text{forget\_gate} = \sigma(W_f \cdot [\text{prev\_state}, \text{input}])
   
   \text{update\_gate} = \sigma(W_u \cdot [\text{prev\_state}, \text{input}])
   
   \text{candidate} = \tanh(W_c \cdot [\text{prev\_state}, \text{input}])
   
   \text{new\_state} = \text{forget\_gate} \odot \text{prev\_state} + \text{update\_gate} \odot \text{candidate}

**Design Choices:**

* **Gating Type**: Support for LSTM, GRU, and custom gates
* **Residual Connections**: Optional skip connections for gradient flow
* **Normalization**: Layer normalization for training stability

State Manager
~~~~~~~~~~~~~

Manages inter-state communication and global state updates:

.. math::

   \text{communication\_matrix} = \text{softmax}(Q \cdot K^T / \sqrt{d_k})
   
   \text{communicated\_states} = \text{communication\_matrix} \cdot V

**Features:**

* **Attention-like Communication**: Optional multi-head attention between states
* **Sparse Communication**: Learnable sparse connectivity patterns
* **Hierarchical States**: Support for multi-level state hierarchies

State-to-Token Projector
~~~~~~~~~~~~~~~~~~~~~~~~

Maps final state representations back to output vocabulary:

.. math::

   \text{token\_affinities} = \text{softmax}(W_o \cdot \text{states})
   
   \text{output\_logits} = \sum_{s} \text{token\_affinities}_{s} \cdot \text{state}_{s}

**Variants:**

* **Direct Projection**: Simple linear transformation
* **Attention-based**: Attention over states for each output position
* **Hybrid**: Combination of both approaches

Architectural Variants
----------------------

Simple NSM
~~~~~~~~~~

The basic NSM implementation with minimal components:

* Fixed number of states (typically 32-128)
* GRU-based state propagation
* Simple routing and projection
* No inter-state communication

**Use Cases:** Text classification, simple sequence modeling

Hybrid NSM
~~~~~~~~~~

Combines NSMs with attention mechanisms:

* NSM layers for efficient long-range modeling
* Transformer layers for complex reasoning
* Flexible layer ordering and configurations
* Shared or separate embeddings

**Use Cases:** Complex reasoning tasks, machine translation

Memory-Augmented NSM
~~~~~~~~~~~~~~~~~~~

Incorporates external memory mechanisms:

* Neural Turing Machine (NTM) memory
* Differentiable memory addressing
* Read/write operations on external memory
* Memory-state interaction layers

**Use Cases:** Algorithmic tasks, long-term memory requirements

Performance Characteristics
---------------------------

Complexity Analysis
~~~~~~~~~~~~~~~~~~

.. list-table:: Complexity Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Operation
     - Transformer
     - NSM
   * - Self-Attention
     - O(n²·d)
     - O(s·d) routing
   * - Feed-Forward
     - O(n·d²)
     - O(s·d²)
   * - Total per Layer
     - O(n²·d + n·d²)
     - O(s·d + s·d²)
   * - Memory Usage
     - O(n²)
     - O(s)

Memory Efficiency
~~~~~~~~~~~~~~~~

NSMs achieve significant memory savings through:

* **Constant State Size**: Fixed memory regardless of sequence length
* **No Attention Matrices**: Eliminates O(n²) attention storage
* **Efficient Routing**: Sparse token-to-state mappings
* **Gradient Efficiency**: Reduced memory for gradient computation

Scalability Properties
~~~~~~~~~~~~~~~~~~~~~

**Sequence Length Scaling:**

* Transformers: Quadratic degradation
* NSMs: Constant or linear scaling
* **Break-even Point**: NSMs become advantageous at ~500+ tokens

**State Count Scaling:**

* **Empirical Finding**: Performance saturates at 64-128 states for most tasks
* **Memory Trade-off**: More states = better capacity but higher constant overhead
* **Task-Dependent**: Simple tasks need fewer states, complex tasks benefit from more

Design Principles
-----------------

Efficiency First
~~~~~~~~~~~~~~~

Every component optimized for computational efficiency:

* Minimal matrix multiplications
* Cache-friendly memory access patterns
* Vectorized operations across states
* Optional CUDA kernel implementations

Interpretability
~~~~~~~~~~~~~~~

Architecture designed for understanding:

* **State Specialization**: States often learn task-specific roles
* **Routing Visualization**: Clear token-to-state assignment patterns
* **Communication Patterns**: Interpretable inter-state interactions
* **Debugging Tools**: Built-in visualization and analysis capabilities

Modularity
~~~~~~~~~

Flexible component composition:

* **Pluggable Components**: Easy to swap routing, propagation, or projection methods
* **Configuration-Driven**: Architecture specified through configuration files
* **Research-Friendly**: Simple to experiment with new variants
* **Production-Ready**: Optimized implementations for deployment

Future Directions
-----------------

Planned Enhancements
~~~~~~~~~~~~~~~~~~~

1. **Adaptive State Allocation**: Dynamic state count based on input complexity
2. **Hierarchical States**: Multi-level state representations
3. **Continuous Learning**: Efficient fine-tuning and adaptation
4. **Hardware Optimization**: Custom CUDA kernels for critical operations

Research Opportunities
~~~~~~~~~~~~~~~~~~~~~

1. **Theoretical Analysis**: Formal complexity and capacity analysis
2. **Architecture Search**: Automated NSM design optimization
3. **Multi-Modal Extensions**: Vision, audio, and cross-modal applications
4. **Large-Scale Evaluation**: Scaling to billions of parameters