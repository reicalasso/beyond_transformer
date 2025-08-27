"""
Tests for configuration management utilities.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nsm.config import Config


def test_config_json():
    """Test Config class with JSON file."""
    # Create a temporary JSON configuration file
    config_data = {
        "model": {
            "input_dim": 784,
            "state_dim": 128,
            "num_states": 16,
            "output_dim": 10,
            "gate_type": "gru"
        },
        "training": {
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Load configuration
        config = Config(config_path)
        
        # Test accessing values
        assert config.get('model.input_dim') == 784
        assert config['training.epochs'] == 5
        assert config.get('model.gate_type') == 'gru'
        
        # Test default values
        assert config.get('model.dropout', 0.1) == 0.1
        assert config.get('nonexistent.key', 'default') == 'default'
        
        # Test setting values
        config['model.dropout'] = 0.2
        assert config['model.dropout'] == 0.2
        
        config.set('training.epochs', 10)
        assert config.get('training.epochs') == 10
        
        # Test validation
        required_keys = ['model.input_dim', 'training.epochs', 'model.state_dim']
        assert config.validate(required_keys) == True
        
        # Test validation with missing key
        required_keys.append('nonexistent.key')
        assert config.validate(required_keys) == False
        
        print("✓ Config JSON test passed")
    finally:
        # Clean up temporary file
        os.unlink(config_path)


def test_config_yaml():
    """Test Config class with YAML file."""
    # Create a temporary YAML configuration file
    config_data = """
model:
  input_dim: 784
  state_dim: 128
  num_states: 16
  output_dim: 10
  gate_type: "gru"

training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_data)
        config_path = f.name
    
    try:
        # Load configuration
        config = Config(config_path)
        
        # Test accessing values
        assert config.get('model.input_dim') == 784
        assert config['training.epochs'] == 5
        assert config.get('model.gate_type') == 'gru'
        
        print("✓ Config YAML test passed")
    finally:
        # Clean up temporary file
        os.unlink(config_path)


def test_config_save():
    """Test saving configuration."""
    # Create a temporary JSON configuration file
    config_data = {
        "model": {
            "input_dim": 784,
            "state_dim": 128
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Load configuration
        config = Config(config_path)
        
        # Modify configuration
        config['model.num_states'] = 16
        config['training.epochs'] = 10
        
        # Save to a new file
        new_config_path = config_path.replace('.json', '_new.json')
        config.save(new_config_path)
        
        # Load the new configuration and verify
        new_config = Config(new_config_path)
        assert new_config['model.num_states'] == 16
        assert new_config['training.epochs'] == 10
        
        # Clean up new file
        os.unlink(new_config_path)
        
        print("✓ Config save test passed")
    finally:
        # Clean up temporary file
        os.unlink(config_path)


def run_all_tests():
    """Run all configuration tests."""
    print("Running Configuration Tests...")
    print("=" * 30)
    
    test_config_json()
    test_config_yaml()
    test_config_save()
    
    print("\n✓ All configuration tests passed!")


if __name__ == "__main__":
    run_all_tests()