#!/usr/bin/env python3
"""
Training script for the PULSE model.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from pulse.config import Config
from pulse.models import SimplePulse
from pulse.data_loaders import get_mnist_dataloaders


def create_model(config):
    """Create model from configuration."""
    model_config = config['model']
    return SimplePulse(
        input_dim=model_config['input_dim'],
        state_dim=model_config['state_dim'],
        num_states=model_config['num_states'],
        output_dim=model_config['output_dim'],
        gate_type=model_config['gate_type']
    )


def train_model(model, train_loader, config):
    """Train the model."""
    training_config = config['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_class = getattr(torch.optim, training_config['optimizer'].capitalize())
    optimizer = optimizer_class(model.parameters(), lr=training_config['learning_rate'])
    
    # Validate configuration
    required_keys = ['training.epochs', 'training.batch_size', 'training.learning_rate']
    if not config.validate(required_keys):
        raise ValueError("Missing required training configuration")
    
    model.train()
    for epoch in range(training_config['epochs']):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Limit training for demonstration
            if batch_idx > 10:
                break
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{training_config['epochs']}: Loss={avg_loss:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train PULSE model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create model
    model = create_model(config)
    print(f"Model created: {model}")
    
    # Get data loaders
    train_loader, _ = get_mnist_dataloaders("data/", batch_size=config['training.batch_size'])
    
    # Train model
    train_model(model, train_loader, config)
    
    print("Training completed!")


if __name__ == "__main__":
    main()