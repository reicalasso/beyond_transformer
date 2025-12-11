"""
Optimizer and Scheduler utilities for PULSE training.

This module provides:
- Custom optimizers optimized for PULSE training
- Learning rate schedulers with warmup support
- Gradient scaling utilities
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PULSEAdamW(Optimizer):
    """
    AdamW optimizer with PULSE-specific optimizations.
    
    Includes:
    - Separate learning rates for state parameters
    - Gradient noise injection for regularization
    - State-aware weight decay
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        state_lr_multiplier: float = 1.0,
        gradient_noise_scale: float = 0.0,
        amsgrad: bool = False,
    ) -> None:
        """
        Initialize PULSEAdamW optimizer.

        Args:
            params: Model parameters.
            lr: Learning rate.
            betas: Adam beta parameters.
            eps: Epsilon for numerical stability.
            weight_decay: Weight decay coefficient.
            state_lr_multiplier: Multiplier for state parameter learning rate.
            gradient_noise_scale: Scale of gradient noise injection.
            amsgrad: Whether to use AMSGrad variant.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            state_lr_multiplier=state_lr_multiplier,
            gradient_noise_scale=gradient_noise_scale,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PULSEAdamW does not support sparse gradients")

                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Add gradient noise if configured
                if group["gradient_noise_scale"] > 0:
                    noise = torch.randn_like(grad) * group["gradient_noise_scale"]
                    noise *= (1.0 / (1.0 + state["step"]) ** 0.55)  # Decay noise over time
                    grad = grad + noise

                # Decoupled weight decay
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if amsgrad:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1

                # Apply state learning rate multiplier if this is a state parameter
                if hasattr(p, "_is_state_param") and p._is_state_param:
                    step_size *= group["state_lr_multiplier"]

                p.addcdiv_(exp_avg, denom, value=-step_size * math.sqrt(bias_correction2))

        return loss


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize WarmupCosineScheduler.

        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
            min_lr_ratio: Minimum learning rate as ratio of initial lr.
            last_epoch: Last epoch number.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            min_factor = self.min_lr_ratio
            factor = min_factor + (1 - min_factor) * cosine_factor
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and linear decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize WarmupLinearScheduler.

        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
            last_epoch: Last epoch number.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            decay_factor = max(0.0, 1 - current_step / max(1, decay_steps))
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupPolynomialScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and polynomial decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize WarmupPolynomialScheduler.

        Args:
            optimizer: Optimizer to schedule.
            warmup_steps: Number of warmup steps.
            total_steps: Total number of training steps.
            power: Power of polynomial decay.
            end_lr: Final learning rate.
            last_epoch: Last epoch number.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            decay_steps = self.total_steps - self.warmup_steps
            current_step = self.last_epoch - self.warmup_steps
            decay_factor = (1 - current_step / max(1, decay_steps)) ** self.power
            return [
                self.end_lr + (base_lr - self.end_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    state_lr_multiplier: float = 1.0,
    **kwargs: Any,
) -> Optimizer:
    """
    Create an optimizer for PULSE training.

    Args:
        model: The model to optimize.
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd', 'pulse_adamw').
        learning_rate: Learning rate.
        weight_decay: Weight decay coefficient.
        betas: Adam beta parameters.
        eps: Epsilon for numerical stability.
        state_lr_multiplier: Multiplier for state parameter learning rate.
        **kwargs: Additional optimizer arguments.

    Returns:
        Configured optimizer.
    """
    # Separate parameters into groups
    decay_params = []
    no_decay_params = []
    state_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this is a state parameter
        is_state = "state" in name.lower() or "initial_states" in name

        if is_state:
            param._is_state_param = True
            state_params.append(param)
        elif "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": state_params, "weight_decay": weight_decay * 0.1, "lr": learning_rate * state_lr_multiplier},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if g["params"]]

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=learning_rate,
            momentum=kwargs.get("momentum", 0.9),
        )
    elif optimizer_type == "pulse_adamw":
        optimizer = PULSEAdamW(
            param_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            state_lr_multiplier=state_lr_multiplier,
            gradient_noise_scale=kwargs.get("gradient_noise_scale", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    warmup_steps: int = 0,
    total_steps: int = 10000,
    **kwargs: Any,
) -> _LRScheduler:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler ('linear', 'cosine', 'polynomial', 'constant').
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.
        **kwargs: Additional scheduler arguments.

    Returns:
        Configured scheduler.
    """
    if scheduler_type == "linear":
        scheduler = WarmupLinearScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    elif scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        )
    elif scheduler_type == "polynomial":
        scheduler = WarmupPolynomialScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            power=kwargs.get("power", 1.0),
            end_lr=kwargs.get("end_lr", 0.0),
        )
    elif scheduler_type == "constant":
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler
