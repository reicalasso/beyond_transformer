"""
Script to summarize experiment results.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pulse.experiment_results import ExperimentResults


def summarize_state_count_sweep(results_manager: ExperimentResults, experiment_name: str = "state_count_sweep"):
    """
    Summarize results from state count sweep experiment.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
        experiment_name (str): Name of the experiment
    """
    try:
        # Load results
        results = results_manager.load_experiment_results(experiment_name)
        results_data = results['results']
        
        # Calculate statistics
        max_accuracy = max(results_data['accuracies'])
        max_accuracy_index = results_data['accuracies'].index(max_accuracy)
        optimal_state_count = results_data['state_counts'][max_accuracy_index]
        
        min_memory = min(results_data['memory_usages'])
        min_memory_index = results_data['memory_usages'].index(min_memory)
        min_memory_state_count = results_data['state_counts'][min_memory_index]
        
        max_memory = max(results_data['memory_usages'])
        max_memory_index = results_data['memory_usages'].index(max_memory)
        max_memory_state_count = results_data['state_counts'][max_memory_index]
        
        # Create summary
        summary = f"""# State Count Sweep Experiment Summary

## Overview
This experiment tested the effect of varying the number of state nodes on model performance.

## Key Findings
- **Optimal State Count**: {optimal_state_count} states achieved maximum accuracy of {max_accuracy:.2f}%
- **Memory Efficiency**: {min_memory_state_count} states used minimum memory ({min_memory:.2f} MB)
- **Memory Intensive**: {max_memory_state_count} states used maximum memory ({max_memory:.2f} MB)

## Detailed Results
| State Count | Accuracy (%) | Memory Usage (MB) | Training Time (s) |
|-------------|--------------|-------------------|-------------------|
"""
        for i in range(len(results_data['state_counts'])):
            summary += f"| {results_data['state_counts'][i]} | {results_data['accuracies'][i]:.2f} | {results_data['memory_usages'][i]:.2f} | {results_data['training_times'][i]:.2f} |\n"
        
        summary += f"""
## Metadata
- Experiment Name: {results['experiment_name']}
- Timestamp: {results['timestamp']}
"""
        
        # Save summary
        filepath = results_manager.save_summary(f"{experiment_name}_summary", summary)
        print(f"Summary saved to: {filepath}")
        
        return summary
    except FileNotFoundError:
        print(f"No results found for experiment: {experiment_name}")
        return None


def summarize_dynamic_allocation(results_manager: ExperimentResults, experiment_name: str = "dynamic_allocation"):
    """
    Summarize results from dynamic allocation experiment.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
        experiment_name (str): Name of the experiment
    """
    try:
        # Load results
        results = results_manager.load_experiment_results(experiment_name)
        results_data = results['results']
        
        # Calculate statistics
        final_accuracy = results_data['accuracies'][-1]
        initial_active_states = results_data['active_states'][0]
        final_active_states = results_data['active_states'][-1]
        total_pruned = sum(results_data['pruned_states'])
        total_allocated = sum(results_data['allocated_states'])
        
        # Create summary
        summary = f"""# Dynamic State Allocation Experiment Summary

## Overview
This experiment evaluated the effect of dynamic state allocation and pruning on model performance.

## Key Findings
- **Final Accuracy**: {final_accuracy:.2f}%
- **Initial Active States**: {initial_active_states}
- **Final Active States**: {final_active_states}
- **Total States Pruned**: {total_pruned}
- **Total States Allocated**: {total_allocated}

## Detailed Results
- **Accuracy Progress**: Started at {results_data['accuracies'][0]:.2f}%, ended at {final_accuracy:.2f}%
- **Active States Range**: {min(results_data['active_states'])} - {max(results_data['active_states'])}
- **Pruning Events**: {len([x for x in results_data['pruned_states'] if x > 0])} batches had pruning
- **Allocation Events**: {len([x for x in results_data['allocated_states'] if x > 0])} batches had allocation

## Metadata
- Experiment Name: {results['experiment_name']}
- Timestamp: {results['timestamp']}
"""
        
        # Save summary
        filepath = results_manager.save_summary(f"{experiment_name}_summary", summary)
        print(f"Summary saved to: {filepath}")
        
        return summary
    except FileNotFoundError:
        print(f"No results found for experiment: {experiment_name}")
        return None


def create_overall_summary(results_manager: ExperimentResults):
    """
    Create an overall summary of all experiments.
    
    Args:
        results_manager (ExperimentResults): Experiment results manager
    """
    experiments = results_manager.list_experiments()
    
    if not experiments:
        print("No experiments found for summary")
        return None
    
    # Create overall summary
    summary = """# Overall Experiment Summary

## Experiments Conducted
"""
    
    for experiment in experiments:
        summary += f"- {experiment}\n"
    
    summary += """
## Next Steps
- Analyze results in detail
- Identify optimal configurations
- Plan follow-up experiments
- Prepare report for stakeholders
"""
    
    # Save summary
    filepath = results_manager.save_summary("overall_summary", summary)
    print(f"Overall summary saved to: {filepath}")
    
    return summary


def main():
    """Main function to summarize all experiment results."""
    # Create experiment results manager
    results_manager = ExperimentResults()
    
    # Summarize state count sweep
    summarize_state_count_sweep(results_manager)
    
    # Summarize dynamic allocation
    summarize_dynamic_allocation(results_manager)
    
    # Create overall summary
    create_overall_summary(results_manager)
    
    print("All summaries generated!")


if __name__ == "__main__":
    main()