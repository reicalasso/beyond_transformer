#!/usr/bin/env python3
"""
Main script to run PULSE experiments
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pulse.experiments import run_state_count_sweep, save_results
from pulse.experiments.dynamic_state_allocation import run_dynamic_state_experiment


def main():
    """Main function to run experiments."""
    print("PULSE Experiments")
    print("=" * 40)
    
    print("\n1. Running State Count Hyperparameter Sweep...")
    results = run_state_count_sweep()
    save_results(results, 'state_count_sweep_results.json')
    
    print("\n2. Running Dynamic State Allocation Experiment...")
    dynamic_results = run_dynamic_state_experiment()
    
    print("\nAll experiments completed successfully!")


if __name__ == "__main__":
    main()