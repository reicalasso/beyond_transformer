"""
PULSE Mixture of Experts and Mixture of Depths

Provides dynamic computation allocation:
- MixtureOfExperts: Route to top-k experts
- MixtureOfDepths: Skip layers for easy tokens
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ffn import SwiGLU


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) layer.
    
    Routes each token to top-k experts based on learned routing.
    
    Args:
        hidden_size: Model dimension
        intermediate_size: Expert FFN dimension
        num_experts: Total number of experts
        top_k: Number of experts to route to
        noise_std: Noise for load balancing during training
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int = None,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            SwiGLU(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing scores
        router_logits = self.router(x)  # [batch, seq, num_experts]
        
        # Add noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Get top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        
        # Normalize
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Process through experts
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]
            expert_prob = top_k_probs[:, :, k:k+1]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e).unsqueeze(-1).float()
                if mask.sum() > 0:
                    expert_out = self.experts[e](x)
                    output = output + mask * expert_prob * expert_out
        
        # Load balancing loss
        expert_usage = router_probs.mean(dim=(0, 1))
        target = 1.0 / self.num_experts
        aux_loss = ((expert_usage - target) ** 2).sum() * 0.01
        
        return output, aux_loss


class MixtureOfDepths(nn.Module):
    """
    Mixture of Depths - adaptive computation.
    
    Learns to skip computation for "easy" tokens.
    Each token gets a capacity score; only top-c% are processed.
    
    Args:
        hidden_size: Model dimension
        capacity_factor: Fraction of tokens to process (0.5 = 50%)
    """
    
    def __init__(
        self,
        hidden_size: int,
        capacity_factor: float = 0.5,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        
        # Router to decide which tokens to process
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        layer_fn: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with selective processing.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            layer_fn: Function to apply to selected tokens
            
        Returns:
            output: [batch, seq_len, hidden_size]
            routing_weights: [batch, seq_len, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing scores
        routing_scores = self.router(x)  # [batch, seq, 1]
        routing_weights = torch.sigmoid(routing_scores)
        
        # During training, use soft routing
        if self.training:
            # Process all tokens but weight by routing
            processed = layer_fn(x)
            output = routing_weights * processed + (1 - routing_weights) * x
        else:
            # During inference, hard routing for efficiency
            capacity = int(seq_len * self.capacity_factor)
            
            # Get top-k tokens to process
            _, top_indices = routing_scores.squeeze(-1).topk(capacity, dim=-1)
            
            # Create output
            output = x.clone()
            
            # Process selected tokens
            for b in range(batch_size):
                selected = x[b, top_indices[b]]  # [capacity, hidden]
                processed = layer_fn(selected.unsqueeze(0)).squeeze(0)
                output[b, top_indices[b]] = processed
        
        return output, routing_weights


class AdaptiveComputation(nn.Module):
    """
    Adaptive Computation Time (ACT).
    
    Dynamically decides how many computation steps each token needs.
    Halts when confidence is high enough.
    
    Args:
        hidden_size: Model dimension
        max_steps: Maximum computation steps
        halt_threshold: Confidence threshold to halt
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_steps: int = 8,
        halt_threshold: float = 0.99,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.halt_threshold = halt_threshold
        
        # Halting probability predictor
        self.halt_predictor = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        step_fn: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with adaptive computation.
        
        Args:
            x: Input [batch, seq_len, hidden_size]
            step_fn: Function for one computation step
            
        Returns:
            output: [batch, seq_len, hidden_size]
            ponder_cost: Computation cost for regularization
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Track halting
        halted = torch.zeros(batch_size, seq_len, 1, device=device)
        remainders = torch.zeros(batch_size, seq_len, 1, device=device)
        n_updates = torch.zeros(batch_size, seq_len, 1, device=device)
        
        # Accumulate output
        output = torch.zeros_like(x)
        
        for step in range(self.max_steps):
            # Compute step
            x = step_fn(x)
            
            # Halting probability
            halt_prob = torch.sigmoid(self.halt_predictor(x))
            
            # Update for non-halted tokens
            still_running = (halted < 1).float()
            
            # Last step: use remainder
            if step == self.max_steps - 1:
                halt_prob = still_running
            
            # Update
            new_halted = halted + halt_prob * still_running
            remainders = remainders + still_running * (1 - halted)
            n_updates = n_updates + still_running
            
            # Accumulate output
            output = output + halt_prob * still_running * x
            
            halted = new_halted
            
            # Check if all halted
            if (halted >= self.halt_threshold).all():
                break
        
        # Ponder cost for regularization
        ponder_cost = remainders.mean()
        
        return output, ponder_cost
