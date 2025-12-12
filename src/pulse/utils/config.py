"""
PULSE Configuration Management

Loads and validates YAML configuration files for training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..models.pulse import PulseConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: Optional[int] = None
    
    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0
    
    # Gradient
    max_grad_norm: float = 1.0
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    
    # Saving
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DataConfig:
    """Data configuration."""
    
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    max_seq_length: int = 512
    preprocessing_num_workers: int = 4
    
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Dataset specific
    max_samples: Optional[int] = None
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OutputConfig:
    """Output configuration."""
    
    output_dir: str = "./output"
    logging_dir: str = "./logs"
    overwrite_output_dir: bool = False
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass 
class FullConfig:
    """Complete configuration combining all sections."""
    
    model: PulseConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": {k: v for k, v in self.training.__dict__.items()},
            "data": {k: v for k, v in self.data.__dict__.items()},
            "output": {k: v for k, v in self.output.__dict__.items()},
            "seed": self.seed,
        }


def load_config(config_path: Union[str, Path]) -> FullConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        FullConfig object with all settings
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    
    # Parse model config
    model_dict = raw_config.get("model", {})
    model_config = PulseConfig.from_dict(model_dict)
    
    # Parse training config
    training_dict = raw_config.get("training", {})
    training_config = TrainingConfig.from_dict(training_dict)
    
    # Parse data config
    data_dict = raw_config.get("data", {})
    data_config = DataConfig.from_dict(data_dict)
    
    # Parse output config
    output_dict = raw_config.get("output", {})
    output_config = OutputConfig.from_dict(output_dict)
    
    # Seed
    seed = raw_config.get("seed", 42)
    
    return FullConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        output=output_config,
        seed=seed,
    )


def merge_configs(base_config: FullConfig, overrides: Dict[str, Any]) -> FullConfig:
    """
    Merge command-line overrides into config.
    
    Args:
        base_config: Base configuration from YAML
        overrides: Dictionary of overrides (flat keys like "model.hidden_size")
        
    Returns:
        Updated FullConfig
    """
    config_dict = base_config.to_dict()
    
    for key, value in overrides.items():
        if value is None:
            continue
            
        parts = key.split(".")
        target = config_dict
        
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        
        target[parts[-1]] = value
    
    # Reconstruct config
    return FullConfig(
        model=PulseConfig.from_dict(config_dict["model"]),
        training=TrainingConfig.from_dict(config_dict["training"]),
        data=DataConfig.from_dict(config_dict["data"]),
        output=OutputConfig.from_dict(config_dict["output"]),
        seed=config_dict.get("seed", 42),
    )
