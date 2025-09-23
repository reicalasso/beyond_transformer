"""
Configuration management utilities.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class Config:
    """
    Configuration management class.

    This class handles loading, validating, and accessing configuration parameters
    from JSON or YAML files.
    """

    def __init__(self, config_path: str):
        """
        Initialize the Config object.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        # Check if file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        # Determine file extension
        _, ext = os.path.splitext(self.config_path)

        # Load based on extension
        with open(self.config_path, "r") as f:
            if ext.lower() in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            elif ext.lower() == ".json":
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key (str): Configuration key (e.g., 'model.input_dim')
            default (Any): Default value if key is not found

        Returns:
            Any: Configuration value
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key (str): Configuration key (e.g., 'model.input_dim')
            value (Any): Value to set
        """
        keys = key.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def save(self, path: str = None) -> None:
        """
        Save configuration to file.

        Args:
            path (str, optional): Path to save to. If None, uses original path.
        """
        save_path = path or self.config_path

        # Determine file extension
        _, ext = os.path.splitext(save_path)

        # Save based on extension
        with open(save_path, "w") as f:
            if ext.lower() in [".yaml", ".yml"]:
                yaml.dump(self.config, f, default_flow_style=False)
            elif ext.lower() == ".json":
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")

    def validate(self, required_keys: list) -> bool:
        """
        Validate that all required keys are present.

        Args:
            required_keys (list): List of required keys (e.g., ['model.input_dim', 'training.epochs'])

        Returns:
            bool: True if all required keys are present, False otherwise
        """
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required configuration key: {key}")
                return False
        return True

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using bracket notation.

        Args:
            key (str): Configuration key

        Returns:
            Any: Configuration value
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using bracket notation.

        Args:
            key (str): Configuration key
            value (Any): Value to set
        """
        self.set(key, value)

    def __str__(self) -> str:
        """
        String representation of the configuration.

        Returns:
            str: JSON representation of the configuration
        """
        return json.dumps(self.config, indent=2)


# Example usage
if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    config_path = os.path.join(project_root, "configs", "default_config.json")

    # Load configuration
    config = Config(config_path)

    # Print configuration
    print("Configuration:")
    print(config)

    # Access configuration values
    print(f"\nModel input dimension: {config.get('model.input_dim')}")
    print(f"Training epochs: {config['training.epochs']}")

    # Set a new value
    config["training.learning_rate"] = 0.0001
    print(f"Updated learning rate: {config['training.learning_rate']}")

    # Save configuration
    # config.save()  # This would overwrite the original file

    print("Configuration management test completed!")
