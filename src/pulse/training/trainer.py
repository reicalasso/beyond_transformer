"""
PULSE Trainer - Main training loop for PULSE models.

This module provides a comprehensive training framework with:
- Mixed precision training (AMP)
- Gradient accumulation
- Distributed training support
- Logging (TensorBoard, WandB)
- Checkpointing and resumption
- Early stopping
- Learning rate scheduling
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """Arguments for PULSE training."""

    # Output
    output_dir: str = "./output"
    overwrite_output_dir: bool = False

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0

    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    lr_scheduler_type: str = "linear"  # linear, cosine, constant, polynomial

    # Mixed precision
    fp16: bool = False
    bf16: bool = False

    # Logging
    logging_dir: str = "./logs"
    logging_steps: int = 100
    logging_first_step: bool = True
    log_level: str = "info"

    # Evaluation
    eval_strategy: str = "steps"  # no, steps, epoch
    eval_steps: int = 500
    eval_delay: int = 0

    # Saving
    save_strategy: str = "steps"  # no, steps, epoch
    save_steps: int = 500
    save_total_limit: Optional[int] = 3
    save_on_each_node: bool = False
    load_best_model_at_end: bool = True

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0

    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None

    # Hardware
    no_cuda: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Distributed
    local_rank: int = -1
    ddp_backend: str = "nccl"

    # Misc
    run_name: Optional[str] = None
    disable_tqdm: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    def __post_init__(self):
        """Post-initialization processing."""
        if self.output_dir:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir:
            self.logging_dir = os.path.expanduser(self.logging_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def save(self, path: str) -> None:
        """Save arguments to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingArguments":
        """Load arguments from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class PulseTrainer:
    """
    Trainer for PULSE models.

    Handles the complete training loop including:
    - Forward/backward passes
    - Gradient accumulation
    - Mixed precision training
    - Logging and checkpointing
    - Evaluation
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model: The model to train.
            args: Training arguments.
            train_dataset: Training dataset.
            eval_dataset: Evaluation dataset.
            data_collator: Data collator function.
            compute_metrics: Function to compute metrics.
            callbacks: List of callback functions.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or self._default_data_collator
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Setup optimizer and scheduler (will be created in train())
        self.optimizer = None
        self.lr_scheduler = None

        # Mixed precision
        self.scaler = GradScaler() if args.fp16 else None
        self.use_amp = args.fp16 or args.bf16
        self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.best_model_checkpoint = None
        self.early_stopping_counter = 0

        # Logging
        self.log_history = []
        self._setup_logging()

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.args.no_cuda or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def _setup_logging(self) -> None:
        """Setup logging."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=getattr(logging, self.args.log_level.upper()),
        )

        # TensorBoard
        if "tensorboard" in self.args.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
            except ImportError:
                logger.warning("TensorBoard not available. Skipping TensorBoard logging.")
                self.tb_writer = None
        else:
            self.tb_writer = None

        # WandB
        if "wandb" in self.args.report_to:
            try:
                import wandb
                wandb.init(
                    project="pulse-training",
                    name=self.args.run_name,
                    config=self.args.to_dict(),
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("WandB not available. Skipping WandB logging.")
                self.wandb = None
        else:
            self.wandb = None

    def _default_data_collator(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Default data collator."""
        batch = {}
        for key in features[0].keys():
            if isinstance(features[0][key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])
            else:
                batch[key] = torch.tensor([f[key] for f in features])
        return batch

    def get_train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=True,
        )

    def get_eval_dataloader(self) -> DataLoader:
        """Create evaluation dataloader."""
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Separate weight decay for different parameter groups
        decay_parameters = []
        no_decay_parameters = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_parameters, "weight_decay": self.args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ]

        if self.args.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        elif self.args.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        elif self.args.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.args.optimizer_type}")

        return optimizer

    def create_scheduler(
        self, optimizer: torch.optim.Optimizer, num_training_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        num_warmup_steps = self.args.warmup_steps
        if self.args.warmup_ratio > 0:
            num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        if self.args.lr_scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps,
            )
        elif self.args.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
            )
        elif self.args.lr_scheduler_type == "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=num_training_steps,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.args.lr_scheduler_type}")

        # Wrap with warmup if needed
        if num_warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[num_warmup_steps],
            )

        return scheduler

    def training_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data.

        Returns:
            Tuple of (loss, metrics_dict)
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass with optional AMP
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model(**batch)
                loss = outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs["loss"]

        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {"loss": loss.item() * self.args.gradient_accumulation_steps}

        return loss, metrics

    def evaluation_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform a single evaluation step.

        Args:
            batch: Batch of data.

        Returns:
            Tuple of (loss, outputs_dict)
        """
        self.model.eval()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad():
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

        return outputs.get("loss", torch.tensor(0.0)), outputs

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=self.args.disable_tqdm):
            loss, outputs = self.evaluation_step(batch)
            batch_size = batch["input_ids"].shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if "logits" in outputs:
                predictions = outputs["logits"].argmax(dim=-1)
                all_predictions.append(predictions.cpu())

            if "labels" in batch:
                all_labels.append(batch["labels"].cpu())

        metrics = {"eval_loss": total_loss / total_samples}

        # Compute custom metrics if provided
        if self.compute_metrics is not None and all_predictions:
            predictions = torch.cat(all_predictions)
            labels = torch.cat(all_labels) if all_labels else None
            custom_metrics = self.compute_metrics((predictions, labels))
            metrics.update(custom_metrics)

        return metrics

    def save_checkpoint(self, output_dir: Optional[str] = None) -> None:
        """Save model checkpoint."""
        output_dir = output_dir or self.args.output_dir
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{self.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

        # Save optimizer and scheduler
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f)

        # Save training arguments
        self.args.save(os.path.join(checkpoint_dir, "training_args.json"))

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Manage checkpoint limit
        if self.args.save_total_limit is not None:
            self._rotate_checkpoints(output_dir)

    def _rotate_checkpoints(self, output_dir: str) -> None:
        """Remove old checkpoints if exceeding limit."""
        checkpoints = []
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, item)))

        checkpoints.sort(key=lambda x: x[0])

        while len(checkpoints) > self.args.save_total_limit:
            _, checkpoint_path = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {checkpoint_path}")
            import shutil
            shutil.rmtree(checkpoint_path)

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load model checkpoint."""
        # Load model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load optimizer
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path) and self.optimizer is not None:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

        # Load scheduler
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))

        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_metric = state.get("best_metric")

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    def log(self, logs: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        step = step or self.global_step

        # Add to history
        logs["step"] = step
        self.log_history.append(logs)

        # Console logging
        log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in logs.items()])
        logger.info(log_str)

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # WandB
        if self.wandb is not None:
            self.wandb.log(logs, step=step)

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Run the training loop.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from.

        Returns:
            Dictionary of training metrics.
        """
        # Set seed
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        # Create dataloader
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.args.num_train_epochs

        # Create optimizer and scheduler
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num epochs = {self.args.num_train_epochs}")
        logger.info(f"  Batch size = {self.args.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_training_steps}")

        # Training loop
        train_loss = 0.0
        start_time = time.time()

        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            epoch_iterator = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=self.args.disable_tqdm,
            )

            for step, batch in enumerate(epoch_iterator):
                # Training step
                loss, metrics = self.training_step(batch)
                train_loss += metrics["loss"]

                # Gradient accumulation
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.args.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.args.max_grad_norm
                        )

                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        avg_loss = train_loss / self.args.logging_steps
                        lr = self.lr_scheduler.get_last_lr()[0]
                        self.log({
                            "train_loss": avg_loss,
                            "learning_rate": lr,
                            "epoch": epoch + step / len(train_dataloader),
                        })
                        train_loss = 0.0

                    # Evaluation
                    if (
                        self.args.eval_strategy == "steps"
                        and self.global_step % self.args.eval_steps == 0
                        and self.global_step >= self.args.eval_delay
                    ):
                        eval_metrics = self.evaluate()
                        self.log(eval_metrics)

                        # Early stopping check
                        if self._check_early_stopping(eval_metrics):
                            logger.info("Early stopping triggered")
                            return self._finalize_training()

                    # Save checkpoint
                    if (
                        self.args.save_strategy == "steps"
                        and self.global_step % self.args.save_steps == 0
                    ):
                        self.save_checkpoint()

            # End of epoch evaluation
            if self.args.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                self.log(eval_metrics)

                if self._check_early_stopping(eval_metrics):
                    logger.info("Early stopping triggered")
                    return self._finalize_training()

            # End of epoch save
            if self.args.save_strategy == "epoch":
                self.save_checkpoint()

        return self._finalize_training()

    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        if self.args.early_stopping_patience is None:
            return False

        # Use eval_loss as the metric to monitor
        current_metric = metrics.get("eval_loss", float("inf"))

        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_model_checkpoint = self.global_step
            return False

        # Check if improved (lower is better for loss)
        if current_metric < self.best_metric - self.args.early_stopping_threshold:
            self.best_metric = current_metric
            self.best_model_checkpoint = self.global_step
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        return self.early_stopping_counter >= self.args.early_stopping_patience

    def _finalize_training(self) -> Dict[str, float]:
        """Finalize training and return metrics."""
        # Load best model if configured
        if self.args.load_best_model_at_end and self.best_model_checkpoint is not None:
            best_checkpoint_dir = os.path.join(
                self.args.output_dir, f"checkpoint-{self.best_model_checkpoint}"
            )
            if os.path.exists(best_checkpoint_dir):
                self.load_checkpoint(best_checkpoint_dir)
                logger.info(f"Loaded best model from {best_checkpoint_dir}")

        # Save final model
        final_dir = os.path.join(self.args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(final_dir, "model.pt"))

        # Close loggers
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb is not None:
            self.wandb.finish()

        return {"best_metric": self.best_metric, "global_step": self.global_step}
