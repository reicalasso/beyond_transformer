#!/usr/bin/env python3
"""
Results Summary for Baseline Comparison Experiment
"""

import json
import numpy as np

# Load results
with open('baseline_comparison_results.json', 'r') as f:
    results = json.load(f)

print("Baseline Model Comparison Results Summary")
print("=" * 50)

# Print results for each dataset
for dataset_name, models in results.items():
    print(f"\n{dataset_name}:")
    print("-" * (len(dataset_name) + 1))
    
    # Filter out models with errors
    successful_models = {k: v for k, v in models.items() if 'error' not in v}
    
    if successful_models:
        print(f"{'Model':<12} {'Train Acc':<10} {'Test Acc':<10} {'Memory (MB)':<12} {'Time (s)':<10}")
        print("-" * 50)
        
        for model_name, metrics in successful_models.items():
            train_acc = metrics['train_accuracy']
            test_acc = metrics['test_accuracy']
            memory = metrics['memory_usage']
            time = metrics['training_time']
            
            print(f"{model_name:<12} {train_acc:<10.2f} {test_acc:<10.2f} {memory:<12.2f} {time:<10.2f}")
    else:
        print("No successful models for this dataset")

print("\n" + "=" * 50)
print("Key Findings:")
print("-" * 15)

# Overall observations
print("1. CIFAR-10: NSM achieved competitive results with fastest training time")
print("2. MNIST: GRU performed best, NSM showed good efficiency")
print("3. Text datasets had implementation issues (data type mismatches)")
print("4. Memory usage varied significantly across models and datasets")
print("5. NSM generally showed efficient training times")

print("\nNote: Text dataset results are incomplete due to implementation issues.")
print("      These will be fixed in future iterations.")
