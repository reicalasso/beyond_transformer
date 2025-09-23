Tutorials
=========

This section contains comprehensive tutorials covering various aspects of Neural State Machines.

.. toctree::
   :maxdepth: 2

   tutorials/basic_usage
   tutorials/custom_models
   tutorials/performance_optimization
   tutorials/visualization
   tutorials/debugging

Overview
--------

These tutorials are designed to take you from basic usage to advanced applications:

1. **Basic Usage**: Learn the fundamentals of NSMs
2. **Custom Models**: Build your own NSM variants
3. **Performance Optimization**: Maximize efficiency and speed
4. **Visualization**: Understand model behavior through visualizations
5. **Debugging**: Tools and techniques for debugging NSMs

Prerequisites
-------------

Before starting these tutorials, ensure you have:

* Completed the :doc:`../quickstart` guide
* Basic understanding of PyTorch
* Familiarity with transformer architectures (helpful but not required)

Tutorial Data
-------------

Some tutorials use example datasets that can be downloaded:

.. code-block:: python

   from nsm.data import download_tutorial_data
   
   # Downloads sample datasets for tutorials
   download_tutorial_data("./tutorial_data")

Getting Help
------------

If you encounter issues with any tutorial:

1. Check the troubleshooting section in each tutorial
2. Review the :doc:`../api/index` documentation
3. Open an issue on GitHub with the ``tutorial`` label