#!/usr/bin/env python3
"""
Visualization script for hyperparameter sweep results
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/experiments/hyperparameter_sweep_results.json', 'r') as f:
    results = json.load(f)

# Extract data
state_counts = results['state_counts']
accuracies = results['accuracies']
test_accuracies = results['test_accuracies']
memory_usages = results['memory_usages']
training_times = results['training_times']

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('NSM State Count Hyperparameter Sweep Results', fontsize=16, fontweight='bold')

# Plot 1: Training Accuracy vs State Count
ax = axes[0, 0]
ax.plot(state_counts, accuracies, marker='o', linewidth=2, markersize=8, color='#1f77b4')
ax.set_xlabel('Number of State Nodes')
ax.set_ylabel('Training Accuracy (%)')
ax.set_title('Training Accuracy vs State Count')
ax.grid(True, alpha=0.3)
ax.set_xticks(state_counts)

# Plot 2: Test Accuracy vs State Count
ax = axes[0, 1]
ax.plot(state_counts, test_accuracies, marker='s', linewidth=2, markersize=8, color='#ff7f0e')
ax.set_xlabel('Number of State Nodes')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy vs State Count')
ax.grid(True, alpha=0.3)
ax.set_xticks(state_counts)

# Plot 3: Memory Usage vs State Count
ax = axes[1, 0]
ax.plot(state_counts, memory_usages, marker='^', linewidth=2, markersize=8, color='#2ca02c')
ax.set_xlabel('Number of State Nodes')
ax.set_ylabel('Memory Usage (MB)')
ax.set_title('Memory Usage vs State Count')
ax.grid(True, alpha=0.3)
ax.set_xticks(state_counts)

# Plot 4: Training Time vs State Count
ax = axes[1, 1]
ax.plot(state_counts, training_times, marker='d', linewidth=2, markersize=8, color='#d62728')
ax.set_xlabel('Number of State Nodes')
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time vs State Count')
ax.grid(True, alpha=0.3)
ax.set_xticks(state_counts)

plt.tight_layout()
plt.savefig('results/visualization/nsm_hyperparameter_sweep_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved to results/visualization/nsm_hyperparameter_sweep_results.png")