Quick Start Guide
================

This guide will get you up and running with Neural State Machines in just a few minutes.

Basic Usage
-----------

Simple Text Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from nsm import SimpleNSM
   
   # Create model for sentiment analysis
   model = SimpleNSM(
       vocab_size=10000,
       d_model=256,
       num_states=64,
       max_seq_length=512,
       num_classes=2  # Binary classification
   )
   
   # Example input (batch_size=2, sequence_length=128)
   input_ids = torch.randint(0, 10000, (2, 128))
   
   # Forward pass
   logits = model(input_ids)  # Shape: [2, 2]
   predictions = torch.softmax(logits, dim=-1)
   
   print(f"Predictions: {predictions}")

Language Modeling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from nsm import SimpleNSM
   
   # Create language model
   model = SimpleNSM(
       vocab_size=50000,
       d_model=512,
       num_states=128,
       max_seq_length=1024,
       is_language_model=True
   )
   
   # Input sequence
   input_ids = torch.randint(0, 50000, (1, 256))
   
   # Generate next token probabilities
   output = model(input_ids)  # Shape: [1, 256, 50000]
   next_token_logits = output[:, -1, :]  # Last position
   
   # Sample next token
   next_token = torch.multinomial(
       torch.softmax(next_token_logits, dim=-1), 
       num_samples=1
   )

Training a Model
----------------

Basic Training Loop
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from torch.optim import Adam
   from nsm import SimpleNSM
   
   # Model setup
   model = SimpleNSM(vocab_size=10000, d_model=256, num_states=64)
   optimizer = Adam(model.parameters(), lr=1e-4)
   criterion = nn.CrossEntropyLoss()
   
   # Training loop
   model.train()
   for epoch in range(10):
       for batch_idx, (data, targets) in enumerate(dataloader):
           optimizer.zero_grad()
           
           # Forward pass
           output = model(data)
           loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
           
           # Backward pass
           loss.backward()
           optimizer.step()
           
           if batch_idx % 100 == 0:
               print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

With Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nsm.utils import PerformanceMonitor
   
   # Create performance monitor
   monitor = PerformanceMonitor()
   
   # Training with monitoring
   model.train()
   for epoch in range(10):
       epoch_start = monitor.start_timing("epoch")
       
       for batch_idx, (data, targets) in enumerate(dataloader):
           # Monitor forward pass
           with monitor.memory_context():
               output = model(data)
               loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
           
           # Monitor backward pass
           with monitor.gradient_context():
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()
       
       epoch_time = monitor.end_timing("epoch")
       
       # Print statistics
       if epoch % 5 == 0:
           stats = monitor.get_stats()
           print(f"Epoch {epoch}: {epoch_time:.2f}s, "
                 f"Memory: {stats['memory']['peak_allocated_mb']:.1f}MB")

Hybrid Models
-------------

Combining NSM with Attention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nsm.models import HybridModel
   
   # Create hybrid model
   model = HybridModel(
       vocab_size=10000,
       d_model=512,
       num_states=64,
       num_attention_layers=2,  # Transformer layers
       num_nsm_layers=4,        # NSM layers
       num_heads=8,
       max_seq_length=1024
   )
   
   # Use like any other model
   input_ids = torch.randint(0, 10000, (2, 512))
   output = model(input_ids)

Configuration Management
------------------------

Using Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nsm.core import NSMConfig
   
   # Load from file
   config = NSMConfig.from_file("config.yaml")
   model = config.build_model()
   
   # Or create programmatically
   config = NSMConfig(
       vocab_size=10000,
       d_model=256,
       num_states=64,
       dropout=0.1,
       learning_rate=1e-4
   )

Example configuration file (``config.yaml``):

.. code-block:: yaml

   # Model architecture
   vocab_size: 10000
   d_model: 256
   num_states: 64
   max_seq_length: 1024
   
   # Training
   learning_rate: 0.0001
   batch_size: 32
   num_epochs: 100
   
   # Regularization
   dropout: 0.1
   weight_decay: 0.01

Benchmarking
------------

Running Standard Benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nsm.benchmarks import LRABenchmark, bAbIBenchmark
   
   # Long Range Arena benchmark
   lra_benchmark = LRABenchmark(
       model_config=config,
       batch_size=32,
       max_epochs=50
   )
   
   lra_results = lra_benchmark.run_all_tasks()
   print(f"LRA Average Score: {lra_results['average']:.2f}%")
   
   # bAbI reasoning tasks
   babi_benchmark = bAbIBenchmark(
       model_config=config,
       tasks=[1, 2, 3, 4, 5]  # Specific tasks
   )
   
   babi_results = babi_benchmark.run_benchmark()
   print(f"bAbI Success Rate: {babi_results['success_rate']:.2f}%")

Model Comparison
~~~~~~~~~~~~~~~

.. code-block:: python

   from nsm.utils import ComparativePerformanceAnalyzer
   
   # Compare multiple models
   analyzer = ComparativePerformanceAnalyzer()
   
   models = {
       'NSM-64': (nsm_model, (batch_size, seq_len)),
       'NSM-128': (larger_nsm_model, (batch_size, seq_len)),
       'Transformer': (transformer_model, (batch_size, seq_len))
   }
   
   results = analyzer.compare_models(models)
   analyzer.generate_comparison_report(results, "model_comparison.md")

Next Steps
----------

* Check out the :doc:`tutorials` for in-depth examples
* Explore the :doc:`api/models` for advanced model configurations
* Learn about :doc:`architecture/performance` optimization techniques
* See :doc:`benchmarks/results` for detailed performance analysis