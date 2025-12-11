"""
Visualization utilities for experiment results.
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pulse.experiment_results import ExperimentResults


def plot_state_count_sweep(results_manager: ExperimentResults, experiment_name: str = "state_count_sweep"):
    """
    Plot results from state count sweep experiment.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
        experiment_name (str): Name of the experiment
    """
    # Load results
    results = results_manager.load_experiment_results(experiment_name)
    df = results_manager.get_experiment_results_df(experiment_name)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'State Count Sweep Results - {results["experiment_name"]}', fontsize=16)
    
    # Plot 1: Accuracy vs State Count
    axes[0, 0].plot(df['state_counts'], df['accuracies'], marker='o')
    axes[0, 0].set_xlabel('Number of States')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Accuracy vs State Count')
    axes[0, 0].grid(True)
    
    # Plot 2: Memory Usage vs State Count
    axes[0, 1].plot(df['state_counts'], df['memory_usages'], marker='s', color='orange')
    axes[0, 1].set_xlabel('Number of States')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_title('Memory Usage vs State Count')
    axes[0, 1].grid(True)
    
    # Plot 3: Training Time vs State Count
    axes[1, 0].plot(df['state_counts'], df['training_times'], marker='^', color='green')
    axes[1, 0].set_xlabel('Number of States')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time vs State Count')
    axes[1, 0].grid(True)
    
    # Plot 4: Accuracy vs Memory Usage (Scatter)
    axes[1, 1].scatter(df['memory_usages'], df['accuracies'], s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Memory Usage (MB)')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy vs Memory Usage')
    axes[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save visualization
    filepath = results_manager.save_visualization(f"{experiment_name}_results", fig)
    print(f"Visualization saved to: {filepath}")
    
    return fig


def plot_dynamic_allocation(results_manager: ExperimentResults, experiment_name: str = "dynamic_allocation"):
    """
    Plot results from dynamic allocation experiment.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
        experiment_name (str): Name of the experiment
    """
    try:
        # Load results
        results = results_manager.load_experiment_results(experiment_name)
        results_data = results['results']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Dynamic Allocation Results - {results["experiment_name"]}', fontsize=16)
        
        # Plot 1: Accuracy over epochs
        epochs = list(range(1, len(results_data['accuracies']) + 1))
        axes[0, 0].plot(epochs, results_data['accuracies'], marker='o')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy (%)')
        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].grid(True)
        
        # Plot 2: Active states over epochs
        axes[0, 1].plot(epochs, results_data['active_states'], marker='s', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Active States')
        axes[0, 1].set_title('Active States Over Time')
        axes[0, 1].grid(True)
        
        # Plot 3: Pruned states over batches
        batch_indices = list(range(len(results_data['pruned_states'])))
        axes[1, 0].plot(batch_indices, results_data['pruned_states'], marker='^', color='red')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].set_ylabel('States Pruned')
        axes[1, 0].set_title('States Pruned per Batch')
        axes[1, 0].grid(True)
        
        # Plot 4: Allocated states over batches
        axes[1, 1].plot(batch_indices, results_data['allocated_states'], marker='d', color='purple')
        axes[1, 1].set_xlabel('Batch')
        axes[1, 1].set_ylabel('States Allocated')
        axes[1, 1].set_title('States Allocated per Batch')
        axes[1, 1].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save visualization
        filepath = results_manager.save_visualization(f"{experiment_name}_results", fig)
        print(f"Visualization saved to: {filepath}")
        
        return fig
    except FileNotFoundError:
        print(f"No results found for experiment: {experiment_name}")
        return None


def create_comparison_plot(results_manager: ExperimentResults):
    """
    Create a comparison plot of different experiments.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
    """
    experiments = results_manager.list_experiments()
    
    if not experiments:
        print("No experiments found for comparison")
        return None
    
    # Collect data from all experiments
    comparison_data = []
    for experiment in experiments:
        try:
            results = results_manager.load_experiment_results(experiment)
            results_data = results['results']
            
            # For state count sweep experiments
            if 'state_counts' in results_data and 'accuracies' in results_data:
                for i in range(len(results_data['state_counts'])):
                    comparison_data.append({
                        'experiment': experiment,
                        'state_count': results_data['state_counts'][i],
                        'accuracy': results_data['accuracies'][i],
                        'memory_usage': results_data['memory_usages'][i],
                        'training_time': results_data['training_times'][i]
                    })
        except FileNotFoundError:
            continue
    
    if not comparison_data:
        print("No comparable data found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Experiment Comparison', fontsize=16)
    
    # Plot 1: Accuracy comparison
    sns.boxplot(data=df, x='experiment', y='accuracy', ax=axes[0])
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Memory usage comparison
    sns.boxplot(data=df, x='experiment', y='memory_usage', ax=axes[1])
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('Memory Usage (MB)')
    axes[1].set_title('Memory Usage Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Training time comparison
    sns.boxplot(data=df, x='experiment', y='training_time', ax=axes[2])
    axes[2].set_xlabel('Experiment')
    axes[2].set_ylabel('Training Time (seconds)')
    axes[2].set_title('Training Time Comparison')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save visualization
    filepath = results_manager.save_visualization("experiment_comparison", fig)
    print(f"Comparison visualization saved to: {filepath}")
    
    return fig


# Example usage
if __name__ == "__main__":
    # Create experiment results manager
    results_manager = ExperimentResults()
    
    # Try to plot state count sweep results
    try:
        fig1 = plot_state_count_sweep(results_manager)
    except FileNotFoundError:
        print("No state count sweep results found")
    
    # Try to plot dynamic allocation results
    try:
        fig2 = plot_dynamic_allocation(results_manager)
    except FileNotFoundError:
        print("No dynamic allocation results found")
    
    # Create comparison plot
    try:
        fig3 = create_comparison_plot(results_manager)
    except Exception as e:
        print(f"Could not create comparison plot: {e}")
    
    print("Visualization test completed!")