"""
Experiment results management utilities.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime


class ExperimentResults:
    """
    Experiment results management class.
    
    This class handles saving, loading, and processing experiment results.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the ExperimentResults object.
        
        Args:
            results_dir (str): Base directory for results
        """
        self.results_dir = Path(results_dir)
        self.experiments_dir = self.results_dir / "experiments"
        self.processed_dir = self.results_dir / "processed"
        self.visualizations_dir = self.results_dir / "visualizations"
        self.logs_dir = self.results_dir / "logs"
        self.summaries_dir = self.results_dir / "summaries"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.results_dir,
            self.experiments_dir,
            self.processed_dir,
            self.visualizations_dir,
            self.logs_dir,
            self.summaries_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_experiment_results(self, experiment_name: str, results: Dict[str, Any], 
                               timestamp: bool = True) -> str:
        """
        Save experiment results to JSON file.
        
        Args:
            experiment_name (str): Name of the experiment
            results (Dict[str, Any]): Experiment results
            timestamp (bool): Whether to add timestamp to filename
            
        Returns:
            str: Path to saved results file
        """
        # Add metadata
        results_with_metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        # Create experiment directory
        experiment_dir = self.experiments_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        if timestamp:
            filename = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"{experiment_name}.json"
        
        # Save results
        filepath = experiment_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        return str(filepath)
    
    def load_experiment_results(self, experiment_name: str, 
                              timestamp: str = None) -> Dict[str, Any]:
        """
        Load experiment results from JSON file.
        
        Args:
            experiment_name (str): Name of the experiment
            timestamp (str, optional): Specific timestamp to load. If None, loads latest.
            
        Returns:
            Dict[str, Any]: Experiment results
        """
        experiment_dir = self.experiments_dir / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Find files
        if timestamp:
            pattern = f"{experiment_name}_{timestamp}*.json"
            files = list(experiment_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No results found for timestamp: {timestamp}")
            filepath = files[0]
        else:
            # Load latest file
            files = list(experiment_dir.glob(f"{experiment_name}_*.json"))
            if not files:
                raise FileNotFoundError(f"No results found for experiment: {experiment_name}")
            filepath = max(files, key=os.path.getctime)
        
        # Load results
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """
        List all experiments.
        
        Returns:
            List[str]: List of experiment names
        """
        experiments = []
        for item in self.experiments_dir.iterdir():
            if item.is_dir():
                experiments.append(item.name)
        return experiments
    
    def save_processed_results(self, name: str, data: Union[Dict, pd.DataFrame]) -> str:
        """
        Save processed results.
        
        Args:
            name (str): Name of the processed results
            data (Union[Dict, pd.DataFrame]): Data to save
            
        Returns:
            str: Path to saved file
        """
        filepath = self.processed_dir / f"{name}.json"
        
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def save_visualization(self, name: str, fig) -> str:
        """
        Save a visualization.
        
        Args:
            name (str): Name of the visualization
            fig: Matplotlib figure object
            
        Returns:
            str: Path to saved file
        """
        filepath = self.visualizations_dir / f"{name}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        return str(filepath)
    
    def save_log(self, name: str, content: str) -> str:
        """
        Save experiment log.
        
        Args:
            name (str): Name of the log
            content (str): Log content
            
        Returns:
            str: Path to saved file
        """
        filepath = self.logs_dir / f"{name}.log"
        with open(filepath, 'w') as f:
            f.write(content)
        return str(filepath)
    
    def save_summary(self, name: str, content: str) -> str:
        """
        Save summary report.
        
        Args:
            name (str): Name of the summary
            content (str): Summary content
            
        Returns:
            str: Path to saved file
        """
        filepath = self.summaries_dir / f"{name}.md"
        with open(filepath, 'w') as f:
            f.write(content)
        return str(filepath)
    
    def get_experiment_results_df(self, experiment_name: str) -> pd.DataFrame:
        """
        Get experiment results as a pandas DataFrame.
        
        Args:
            experiment_name (str): Name of the experiment
            
        Returns:
            pd.DataFrame: Experiment results as DataFrame
        """
        results = self.load_experiment_results(experiment_name)
        results_data = results['results']
        
        # Convert to DataFrame
        if isinstance(results_data, dict):
            # If results is a dict of lists/arrays, create DataFrame directly
            return pd.DataFrame(results_data)
        elif isinstance(results_data, list):
            # If results is a list of dicts, create DataFrame from list
            return pd.DataFrame(results_data)
        else:
            # For other cases, create a single-row DataFrame
            return pd.DataFrame([results_data])


# Example usage
if __name__ == "__main__":
    # Create experiment results manager
    results_manager = ExperimentResults()
    
    # Example results
    example_results = {
        "state_counts": [8, 16, 32, 64],
        "accuracies": [53.2, 65.8, 72.1, 75.3],
        "memory_usages": [100.5, 200.1, 400.3, 800.7],
        "training_times": [120.5, 240.1, 480.3, 960.7]
    }
    
    # Save results
    filepath = results_manager.save_experiment_results("state_count_sweep", example_results)
    print(f"Results saved to: {filepath}")
    
    # Load results
    loaded_results = results_manager.load_experiment_results("state_count_sweep")
    print(f"Loaded results: {loaded_results['experiment_name']}")
    
    # Get as DataFrame
    df = results_manager.get_experiment_results_df("state_count_sweep")
    print(f"Results DataFrame:\n{df}")
    
    # List experiments
    experiments = results_manager.list_experiments()
    print(f"Experiments: {experiments}")
    
    print("Experiment results management test completed!")