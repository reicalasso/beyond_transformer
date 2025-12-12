"""
PULSE Spiking Neural Network Components

Implements biologically-inspired spiking mechanisms:
- Spike-timing dependent plasticity (STDP)
- Pulse-based processing
- Graded neuron activations
- Dynamic resource allocation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron with graded output.
    
    Unlike binary spikes, outputs graded activations based on
    membrane potential, more similar to biological neurons.
    """
    
    def __init__(
        self,
        hidden_size: int,
        threshold: float = 1.0,
        decay: float = 0.9,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.decay = decay
        self.noise_std = noise_std
        
        # Learnable threshold and decay per neuron
        self.threshold_param = nn.Parameter(torch.ones(hidden_size) * threshold)
        self.decay_param = nn.Parameter(torch.ones(hidden_size) * decay)
        
        # Membrane potential (state)
        self.register_buffer('membrane', torch.zeros(1, hidden_size))
    
    def forward(
        self,
        input_current: torch.Tensor,
        reset_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through spiking neuron.
        
        Args:
            input_current: [batch, seq_len, hidden_size] or [batch, hidden_size]
            reset_state: Whether to reset membrane potential
            
        Returns:
            output: Graded spike output
            membrane: Updated membrane potential
        """
        if reset_state:
            self.membrane = torch.zeros(1, self.hidden_size, device=input_current.device)
        
        batch_size = input_current.shape[0]
        has_seq = input_current.dim() == 3
        
        if has_seq:
            seq_len = input_current.shape[1]
            outputs = []
            membrane = self.membrane.expand(batch_size, -1).clone()
            
            for t in range(seq_len):
                current = input_current[:, t, :]
                membrane, output = self._step(membrane, current)
                outputs.append(output)
            
            self.membrane = membrane.mean(dim=0, keepdim=True)
            return torch.stack(outputs, dim=1), membrane
        else:
            membrane = self.membrane.expand(batch_size, -1).clone()
            membrane, output = self._step(membrane, input_current)
            self.membrane = membrane.mean(dim=0, keepdim=True)
            return output, membrane
    
    def _step(
        self,
        membrane: torch.Tensor,
        input_current: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single timestep of LIF dynamics."""
        # Decay membrane potential
        decay = torch.sigmoid(self.decay_param)
        membrane = membrane * decay
        
        # Add input current
        membrane = membrane + input_current
        
        # Add noise for natural variation
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(membrane) * self.noise_std
            membrane = membrane + noise
        
        # Graded output based on membrane potential
        threshold = F.softplus(self.threshold_param)
        
        # Soft spike: smooth activation instead of hard threshold
        output = torch.sigmoid((membrane - threshold) * 5)
        
        # Soft reset: reduce membrane after spike
        reset = output * membrane * 0.5
        membrane = membrane - reset
        
        return membrane, output


class PulseProcessor(nn.Module):
    """
    Pulse-based information processing.
    
    Instead of continuous processing, information is processed
    in discrete "pulses" - similar to how the brain processes
    information in bursts rather than continuously.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_pulses: int = 4,
        pulse_width: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_pulses = num_pulses
        self.pulse_width = pulse_width
        
        # Pulse generators
        self.pulse_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_pulses)
        ])
        
        # Pulse processors
        self.pulse_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for _ in range(num_pulses)
        ])
        
        # Pulse timing (learnable delays)
        self.pulse_delays = nn.Parameter(torch.linspace(0, 1, num_pulses))
        
        # Output combination
        self.output_proj = nn.Linear(hidden_size * num_pulses, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through multiple pulses.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            Pulse-processed output
        """
        batch_size, seq_len, _ = x.shape
        
        pulse_outputs = []
        
        for i, (gate, processor) in enumerate(zip(self.pulse_gates, self.pulse_processors)):
            # Determine pulse activation
            pulse_strength = gate(x)  # [batch, seq_len, 1]
            
            # Process with this pulse (no shifting to avoid dimension issues)
            processed = processor(x)
            
            # Gate the output
            pulse_output = processed * pulse_strength
            pulse_outputs.append(pulse_output)
        
        # Combine all pulses
        combined = torch.cat(pulse_outputs, dim=-1)
        output = self.output_proj(combined)
        
        return output


class STDPLayer(nn.Module):
    """
    Spike-Timing Dependent Plasticity (STDP) inspired layer.
    
    Implements Hebbian-like learning where connection strength
    depends on the relative timing of pre and post-synaptic activity.
    """
    
    def __init__(
        self,
        hidden_size: int,
        stdp_window: int = 10,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.stdp_window = stdp_window
        self.lr = learning_rate
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.02)
        
        # Eligibility traces (for STDP)
        self.register_buffer('pre_trace', torch.zeros(1, hidden_size))
        self.register_buffer('post_trace', torch.zeros(1, hidden_size))
        
        # Trace decay
        self.trace_decay = 0.9
    
    def forward(
        self,
        pre_activity: torch.Tensor,
        post_activity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional STDP update.
        
        Args:
            pre_activity: [batch, seq_len, hidden_size] pre-synaptic activity
            post_activity: Optional post-synaptic activity for STDP
            
        Returns:
            Synaptic output
        """
        # Standard forward pass
        output = F.linear(pre_activity, self.weights)
        
        # Update eligibility traces
        if self.training:
            self._update_traces(pre_activity, post_activity or output)
        
        return output
    
    def _update_traces(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
    ):
        """Update STDP eligibility traces."""
        # Decay existing traces
        self.pre_trace = self.pre_trace * self.trace_decay
        self.post_trace = self.post_trace * self.trace_decay
        
        # Add new activity
        pre_mean = pre.mean(dim=(0, 1)) if pre.dim() == 3 else pre.mean(dim=0)
        post_mean = post.mean(dim=(0, 1)) if post.dim() == 3 else post.mean(dim=0)
        
        self.pre_trace = self.pre_trace + pre_mean.unsqueeze(0)
        self.post_trace = self.post_trace + post_mean.unsqueeze(0)
    
    def stdp_update(self):
        """Apply STDP weight update based on traces."""
        if not self.training:
            return
        
        # LTP: pre before post (strengthen)
        ltp = torch.outer(self.post_trace.squeeze(), self.pre_trace.squeeze())
        
        # LTD: post before pre (weaken)
        ltd = torch.outer(self.pre_trace.squeeze(), self.post_trace.squeeze())
        
        # Update weights
        delta = self.lr * (ltp - ltd * 0.5)
        self.weights.data = self.weights.data + delta.clamp(-0.1, 0.1)


class DynamicRouter(nn.Module):
    """
    Dynamic resource allocation through routing.
    
    Like the brain, only activates relevant modules based on input,
    saving computation and enabling specialization.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert modules
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.load_balance_coef = 0.01
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route input to top-k experts.
        
        Args:
            x: [batch, seq_len, hidden_size]
            
        Returns:
            output: Expert-processed output
            aux_loss: Load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing scores
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise
        
        # Get top-k experts
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Process through selected experts
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [batch, seq_len]
            expert_prob = top_k_probs[:, :, k:k+1]  # [batch, seq_len, 1]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e).unsqueeze(-1).float()
                if mask.sum() > 0:
                    expert_out = self.experts[e](x)
                    output = output + mask * expert_prob * expert_out
        
        # Compute load balancing loss
        # Encourage uniform expert usage
        expert_usage = router_probs.mean(dim=(0, 1))  # [num_experts]
        target_usage = 1.0 / self.num_experts
        aux_loss = self.load_balance_coef * ((expert_usage - target_usage) ** 2).sum()
        
        return output, aux_loss


class NaturalVariation(nn.Module):
    """
    Adds natural variation to outputs to reduce "robotic" feel.
    
    Implements:
    - Controlled noise injection
    - Temperature-based sampling variation
    - Temporal consistency (not random noise each step)
    """
    
    def __init__(
        self,
        hidden_size: int,
        variation_scale: float = 0.05,
        temporal_smoothing: float = 0.9,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variation_scale = variation_scale
        self.temporal_smoothing = temporal_smoothing
        
        # Learnable variation patterns
        self.variation_patterns = nn.Parameter(torch.randn(8, hidden_size) * 0.1)
        
        # Pattern selector
        self.pattern_selector = nn.Linear(hidden_size, 8)
        
        # Persistent noise state for temporal consistency
        self.register_buffer('noise_state', torch.zeros(1, hidden_size))
    
    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Add natural variation to input.
        
        Args:
            x: [batch, seq_len, hidden_size]
            temperature: Controls variation amount
            
        Returns:
            Input with natural variation added
        """
        if not self.training and temperature == 0:
            return x
        
        batch_size, seq_len, _ = x.shape
        
        # Select variation patterns based on input
        pattern_weights = F.softmax(self.pattern_selector(x) * temperature, dim=-1)
        
        # Combine patterns
        variation = torch.matmul(pattern_weights, self.variation_patterns)
        
        # Add temporally smooth noise
        noise_state = self.noise_state.expand(batch_size, -1)
        outputs = []
        
        for t in range(seq_len):
            # Update noise state with temporal smoothing
            new_noise = torch.randn_like(noise_state) * self.variation_scale * temperature
            noise_state = self.temporal_smoothing * noise_state + (1 - self.temporal_smoothing) * new_noise
            
            # Combine structured variation with smooth noise
            total_variation = variation[:, t, :] + noise_state
            outputs.append(x[:, t, :] + total_variation)
        
        self.noise_state = noise_state.mean(dim=0, keepdim=True)
        
        return torch.stack(outputs, dim=1)
