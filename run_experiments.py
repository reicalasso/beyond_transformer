#!/usr/bin/env python3
"""
Main script to run NSM experiments
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nsm.experiments import run_state_count_sweep, save_results


def main():
    """Main function to run experiments."""
    print("Neural State Machine Experiments")
    print("=" * 40)
    
    # Run state count sweep
    results = run_state_count_sweep()
    
    # Save results
    save_results(results)
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()