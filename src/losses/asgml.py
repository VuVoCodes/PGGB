"""
Asynchronous Staleness Guided Multimodal Learning (ASGML) Loss and Scheduler.

This module implements the core ASGML mechanism with two modes:

1. FREQUENCY MODE (simpler):
   - Dominant modality updates every k steps (k > 1)
   - Weak modality updates every step
   - Gradients are always computed at current parameters

2. STALENESS MODE (full contribution):
   - Dominant modality uses gradients computed τ steps ago
   - Gradients are stored in staleness buffer and applied with delay
   - Provides implicit regularization effect

CRITICAL SAFETY RULES:
- Staleness buffer is managed separately from optimizer state
- Fusion head ALWAYS updates every step regardless of modality schedules
- Probe signals drive adaptation but probes never backprop into encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque


class StalenessBuffer:
    """
    Buffer for storing gradient snapshots to enable true staleness.

    True staleness means applying gradients computed at old parameters
    to current parameters:
        θ_{t+τ+1} = θ_{t+τ} - η ∇_{θ_t} L

    This is different from reduced frequency which uses fresh gradients
    applied less often:
        θ_{t+1} = θ_t - η ∇_{θ_t} L  (but only every k steps)
    """

    def __init__(self, modalities: List[str], max_staleness: int = 8):
        """
        Args:
            modalities: List of modality names
            max_staleness: Maximum staleness τ (buffer size per modality)
        """
        self.modalities = modalities
        self.max_staleness = max_staleness

        # Buffer stores gradient dicts keyed by step
        # Each entry: {param_name: grad_tensor}
        self.buffers: Dict[str, deque] = {
            m: deque(maxlen=max_staleness) for m in modalities
        }

        # Track which step each gradient was computed at
        self.step_indices: Dict[str, deque] = {
            m: deque(maxlen=max_staleness) for m in modalities
        }

    def store_gradients(
        self,
        modality: str,
        parameters: Dict[str, nn.Parameter],
        step: int,
    ) -> None:
        """
        Store current gradients for a modality.

        Args:
            modality: Modality name
            parameters: Dict of named parameters (from model.named_parameters())
            step: Current training step
        """
        grad_snapshot = {}
        for name, param in parameters.items():
            if param.grad is not None:
                # Clone and detach to create independent copy
                grad_snapshot[name] = param.grad.clone().detach()

        self.buffers[modality].append(grad_snapshot)
        self.step_indices[modality].append(step)

    def get_stale_gradients(
        self,
        modality: str,
        staleness: int,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve gradients from τ steps ago.

        Args:
            modality: Modality name
            staleness: How many steps old the gradient should be (τ)

        Returns:
            Dict of gradients, or None if not enough history
        """
        buffer = self.buffers[modality]

        if len(buffer) < staleness:
            return None

        # Get gradient from staleness steps ago
        # Index -staleness gives us the gradient from τ steps back
        idx = -staleness
        if abs(idx) <= len(buffer):
            return buffer[idx]
        return None

    def clear(self, modality: Optional[str] = None) -> None:
        """Clear buffer for one or all modalities."""
        if modality is not None:
            self.buffers[modality].clear()
            self.step_indices[modality].clear()
        else:
            for m in self.modalities:
                self.buffers[m].clear()
                self.step_indices[m].clear()

    def __len__(self) -> int:
        """Return minimum buffer length across modalities."""
        return min(len(b) for b in self.buffers.values())


class LearningDynamicsTracker:
    """
    Tracks learning dynamics signals for adaptive staleness/frequency.

    Two signals are monitored:
    1. Gradient magnitude ratio: instantaneous dominance signal
    2. Loss descent rate: temporal learning speed signal

    These signals determine which modality is dominant and by how much.
    """

    def __init__(
        self,
        modalities: List[str],
        window_size: int = 10,
        beta: float = 0.5,
        ema_alpha: float = 0.1,
    ):
        """
        Args:
            modalities: List of modality names
            window_size: Number of steps for loss descent calculation
            beta: Weight for combining gradient (β) vs loss (1-β) signals
            ema_alpha: EMA smoothing factor for gradient norms
        """
        self.modalities = modalities
        self.window_size = window_size
        self.beta = beta
        self.ema_alpha = ema_alpha

        # Loss history for descent rate
        self.loss_history: Dict[str, deque] = {
            m: deque(maxlen=window_size) for m in modalities
        }

        # EMA of gradient norms (more stable than raw values)
        self.grad_norm_ema: Dict[str, float] = {m: 0.0 for m in modalities}

        # Track if we have enough data
        self.steps_tracked = 0

    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Dict[str, float],
    ) -> None:
        """
        Update tracking with current step's losses and gradient norms.

        Args:
            losses: Dict mapping modality names to loss values
            grad_norms: Dict mapping modality names to gradient L2 norms
        """
        for m in self.modalities:
            # Update loss history
            if m in losses:
                self.loss_history[m].append(losses[m])

            # Update gradient norm EMA
            if m in grad_norms:
                old_ema = self.grad_norm_ema[m]
                new_val = grad_norms[m]
                self.grad_norm_ema[m] = (
                    self.ema_alpha * new_val + (1 - self.ema_alpha) * old_ema
                )

        self.steps_tracked += 1

    def compute_learning_speed(self) -> Dict[str, float]:
        """
        Compute learning speed score for each modality.

        Score > 1: modality is learning faster than average (dominant)
        Score < 1: modality is learning slower than average (weak)
        Score = 1: modality is at average pace

        Returns:
            Dict mapping modality names to learning speed scores
        """
        if self.steps_tracked < 2:
            return {m: 1.0 for m in self.modalities}

        # Compute gradient magnitude ratios
        grad_ratios = self._compute_gradient_ratios()

        # Compute loss descent ratios
        loss_ratios = self._compute_loss_descent_ratios()

        # Combine with weighting β
        learning_speeds = {}
        for m in self.modalities:
            g_ratio = grad_ratios.get(m, 1.0)
            l_ratio = loss_ratios.get(m, 1.0)
            learning_speeds[m] = self.beta * g_ratio + (1 - self.beta) * l_ratio

        return learning_speeds

    def _compute_gradient_ratios(self) -> Dict[str, float]:
        """Compute gradient magnitude relative to mean across modalities."""
        emas = self.grad_norm_ema
        mean_norm = sum(emas.values()) / max(len(emas), 1)

        if mean_norm < 1e-8:
            return {m: 1.0 for m in self.modalities}

        return {m: emas[m] / mean_norm for m in self.modalities}

    def _compute_loss_descent_ratios(self) -> Dict[str, float]:
        """
        Compute loss descent rate relative to mean across modalities.

        Positive descent = loss decreasing = learning
        Negative descent = loss increasing = possibly overfitting

        We use absolute value of descent to capture learning activity
        regardless of direction. A modality with increasing loss is still
        "active" and shouldn't have its signal zeroed out.
        """
        descents = {}
        for m in self.modalities:
            history = list(self.loss_history[m])
            if len(history) < 2:
                descents[m] = 0.0
            else:
                # Use absolute descent to capture learning activity
                # This preserves signal even when loss is increasing (overfitting phase)
                raw_descent = history[0] - history[-1]
                descents[m] = abs(raw_descent)

        mean_descent = sum(descents.values()) / max(len(descents), 1)

        if mean_descent < 1e-8:
            return {m: 1.0 for m in self.modalities}

        return {m: d / mean_descent for m, d in descents.items()}

    def get_dominant_modality(self) -> Optional[str]:
        """Return modality with highest learning speed."""
        speeds = self.compute_learning_speed()
        if not speeds:
            return None
        return max(speeds, key=speeds.get)

    def reset(self) -> None:
        """Reset all tracking state."""
        for m in self.modalities:
            self.loss_history[m].clear()
            self.grad_norm_ema[m] = 0.0
        self.steps_tracked = 0


class ASGMLScheduler:
    """
    Schedules modality updates based on learning dynamics.

    Supports three modes:
    - FREQUENCY: Skip updates for dominant modality (every k steps)
    - STALENESS: Apply stale gradients to dominant modality
    - CONTINUOUS: Probe-guided continuous gradient scaling (smooth modulation)

    The scheduler can be:
    - FIXED: Use predetermined ratios/staleness values
    - ADAPTIVE: Adjust based on probe signals or learning dynamics
    """

    def __init__(
        self,
        modalities: List[str],
        mode: str = "frequency",  # "frequency", "staleness", or "continuous"
        adaptation: str = "fixed",  # "fixed" or "adaptive"
        # Fixed mode parameters
        fixed_ratio: int = 2,  # Dominant updates every `fixed_ratio` steps
        fixed_staleness: int = 2,  # Staleness τ for dominant modality
        # Adaptive mode parameters
        tau_base: float = 1.0,
        tau_min: float = 1.0,
        tau_max: float = 8.0,
        threshold_delta: float = 0.1,  # Utilization gap threshold
        # General
        beta: float = 0.5,
        lambda_comp: float = 0.1,  # Gradient compensation factor
        max_staleness_ratio: float = 3.0,  # κ: max ratio between modality staleness values
        continuous_scale_ema: float = 0.3,  # μ: EMA coefficient for boost-scale smoothing
    ):
        """
        Args:
            modalities: List of modality names
            mode: "frequency" (skip updates), "staleness" (use old gradients),
                  or "continuous" (probe-guided gradient scaling)
            adaptation: "fixed" (predetermined) or "adaptive" (probe-driven)
            fixed_ratio: For fixed frequency mode, update dominant every k steps
            fixed_staleness: For fixed staleness mode, use τ-step-old gradients
            tau_base: Baseline for adaptive staleness computation
            tau_min: Minimum staleness/frequency (1 = update every step)
            tau_max: Maximum staleness/frequency bound
            threshold_delta: Utilization gap that triggers adaptation
            beta: Weight for gradient vs loss signals in dynamics tracking
            lambda_comp: Gradient scaling compensation factor
            max_staleness_ratio: κ (kappa) - maximum allowed ratio τ_i/τ_j between
                any two modalities. Prevents "Stale Fusion" problem where one encoder
                drifts too far ahead of others. Set to inf to disable constraint.
        """
        self.modalities = modalities
        self.mode = mode
        self.adaptation = adaptation
        self.fixed_ratio = fixed_ratio
        self.fixed_staleness = fixed_staleness
        self.tau_base = tau_base
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.threshold_delta = threshold_delta
        self.lambda_comp = lambda_comp
        self.max_staleness_ratio = max_staleness_ratio

        # Current staleness/frequency values per modality
        self.current_tau: Dict[str, float] = {m: 1.0 for m in modalities}

        # Learning dynamics tracker
        self.dynamics = LearningDynamicsTracker(
            modalities=modalities,
            beta=beta,
        )

        # Staleness buffer (only used in staleness mode)
        self.staleness_buffer = StalenessBuffer(
            modalities=modalities,
            max_staleness=int(tau_max) + 2,
        )

        # Step counter
        self.current_step = 0

        # Cache dominant modality (updated by external probe signals)
        self._dominant_modality: Optional[str] = None

        # Continuous mode: EMA-smoothed gradient scales per modality
        self.current_continuous_scales: Dict[str, float] = {m: 1.0 for m in modalities}
        self.continuous_scale_ema: float = continuous_scale_ema

    def set_dominant_modality(self, modality: str) -> None:
        """
        Set which modality is dominant (from probe signals).

        In adaptive mode, this is called with probe evaluation results.
        """
        self._dominant_modality = modality

    def update_from_utilization(
        self,
        utilization_scores: Dict[str, float],
        utilization_gap: float,
    ) -> None:
        """
        Update staleness values based on probe utilization signals.

        Args:
            utilization_scores: Probe accuracy per modality
            utilization_gap: Max - min probe accuracy
        """
        if self.adaptation != "adaptive":
            return

        if utilization_gap < self.threshold_delta:
            # Gap is small, no need for strong intervention
            self.current_tau = {m: self.tau_min for m in self.modalities}
            return

        # Scale tau based on utilization - higher utilization = higher staleness
        mean_util = sum(utilization_scores.values()) / len(utilization_scores)

        for m in self.modalities:
            relative_util = utilization_scores[m] / max(mean_util, 1e-8)
            raw_tau = self.tau_base * relative_util
            self.current_tau[m] = max(self.tau_min, min(self.tau_max, raw_tau))

    def get_update_mask(self) -> Dict[str, bool]:
        """
        Determine which modalities should update their encoders this step.

        In frequency mode: dominant modality skips some steps
        In staleness mode: all modalities "update" but dominant uses stale grads

        Returns:
            Dict mapping modality names to whether they should update
        """
        if self.mode in ("staleness", "continuous"):
            # In staleness/continuous mode, all modalities always update
            # Staleness: dominant uses stale gradients (handled separately)
            # Continuous: gradients are scaled, not skipped
            return {m: True for m in self.modalities}

        # Frequency mode: check if each modality should update
        update_mask = {}

        if self.adaptation == "fixed":
            for m in self.modalities:
                # Fixed frequency: dominant updates every fixed_ratio steps
                if m == self._dominant_modality:
                    update_mask[m] = (self.current_step % self.fixed_ratio) == 0
                else:
                    update_mask[m] = True
        else:
            # Adaptive frequency: use current_tau with constraint
            raw_tau = {m: int(round(self.current_tau[m])) for m in self.modalities}
            constrained_tau = self._apply_staleness_constraint(raw_tau)

            for m in self.modalities:
                tau = max(constrained_tau[m], 1)
                update_mask[m] = (self.current_step % tau) == 0

        return update_mask

    def get_staleness_values(self) -> Dict[str, int]:
        """
        Get staleness τ for each modality (for staleness mode).

        Applies relative staleness constraint to prevent Stale Fusion problem.

        Returns:
            Dict mapping modality names to staleness values
        """
        if self.mode != "staleness":
            return {m: 0 for m in self.modalities}

        staleness = {}
        for m in self.modalities:
            if self.adaptation == "fixed":
                if m == self._dominant_modality:
                    staleness[m] = self.fixed_staleness
                else:
                    staleness[m] = 0  # Fresh gradients
            else:
                staleness[m] = int(round(self.current_tau[m])) - 1  # τ-1 steps old

        # Apply relative staleness constraint (Stale Fusion mitigation)
        return self._apply_staleness_constraint(staleness)

    def get_gradient_scales(self) -> Dict[str, float]:
        """
        Compute gradient scaling factors for staleness compensation.

        Standard async SGD theory suggests reducing effective learning rate
        for stale gradients: η_effective = η / (1 + τ)

        We implement this as gradient scaling: scale = 1 / (1 + λ * τ)
        where λ controls the reduction strength.

        With λ=0.1 and τ=2: scale = 1/(1+0.2) = 0.83 (17% reduction)
        With λ=0.1 and τ=4: scale = 1/(1+0.4) = 0.71 (29% reduction)

        Returns:
            Dict mapping modality names to gradient scale factors
        """
        scales = {}
        staleness_vals = self.get_staleness_values()

        for m in self.modalities:
            tau = staleness_vals.get(m, 0)
            # Scale = 1 / (1 + λ * τ) (reduce stale gradient contribution)
            # This follows async SGD convergence theory
            scales[m] = 1.0 / (1.0 + self.lambda_comp * tau)

        return scales

    def get_continuous_scales(
        self,
        utilization_scores: Dict[str, float],
        alpha: float = 0.5,
        scale_max: float = 2.0,
    ) -> Dict[str, float]:
        """
        Compute continuous gradient scaling factors from probe utilization scores.

        Boosts weak modality gradients proportional to the utilization gap while
        leaving the dominant modality unchanged. This is the continuous analog of
        ASGML's frequency mode (which gives more updates to weak modalities).

        Parameters
        ----------
        utilization_scores : Dict[str, float]
            Probe accuracy per modality (from ProbeManager.get_utilization_scores()).
            Higher values indicate more dominant/better-utilized modalities.
        alpha : float
            Boost strength. Controls how aggressively weak modalities are
            boosted. 0.0 = no boost, 1.0 = maximum boost (up to scale_max).
        scale_max : float
            Maximum gradient scale for the weakest modality.
            Prevents excessive gradient amplification.

        Returns
        -------
        Dict[str, float]
            Gradient scale factor per modality. Values in [1.0, scale_max].
            Dominant modalities get 1.0, weak modalities get boosted.
        """
        if not utilization_scores:
            return {m: 1.0 for m in self.modalities}

        max_score = max(utilization_scores.values())
        min_score = min(utilization_scores.values())
        util_gap = max_score - min_score

        scales = {}
        for m in self.modalities:
            if util_gap < 1e-8:
                scales[m] = 1.0
            else:
                rel_weakness = 1.0 - (utilization_scores[m] - min_score) / util_gap
                raw_scale = 1.0 + alpha * rel_weakness
                scales[m] = min(raw_scale, scale_max)

        return scales

    def update_continuous_scales(
        self,
        new_scales: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Update continuous gradient scales with EMA smoothing.

        EMA smoothing prevents abrupt scale changes between probe evaluation
        intervals, which can destabilize training.

        Parameters
        ----------
        new_scales : Dict[str, float]
            New target scales from get_continuous_scales().

        Returns
        -------
        Dict[str, float]
            EMA-smoothed current scales.
        """
        ema = self.continuous_scale_ema
        for m in self.modalities:
            old = self.current_continuous_scales[m]
            new = new_scales.get(m, 1.0)
            self.current_continuous_scales[m] = ema * new + (1 - ema) * old
        return self.current_continuous_scales.copy()

    def step(self) -> None:
        """Increment step counter (call at end of each training step)."""
        self.current_step += 1

    def _apply_staleness_constraint(
        self, staleness_values: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Apply relative staleness constraint to prevent Stale Fusion problem.

        Ensures τ_i / τ_j ≤ κ for all modality pairs (i, j).
        This bounds the "training age gap" between encoders.

        Args:
            staleness_values: Raw staleness values per modality

        Returns:
            Constrained staleness values
        """
        if self.max_staleness_ratio <= 0 or self.max_staleness_ratio == float('inf'):
            return staleness_values

        # Find minimum staleness (modality learning slowest, needs most updates)
        min_tau = min(max(v, 1) for v in staleness_values.values())

        # Constrain all modalities: τ_i ≤ κ * min_τ
        constrained = {}
        for m, tau in staleness_values.items():
            max_allowed = int(min_tau * self.max_staleness_ratio)
            constrained[m] = min(tau, max_allowed)

        return constrained

    def reset(self) -> None:
        """Reset scheduler state (call at start of training)."""
        self.current_step = 0
        self.current_tau = {m: 1.0 for m in self.modalities}
        self.current_continuous_scales = {m: 1.0 for m in self.modalities}
        self.dynamics.reset()
        self.staleness_buffer.clear()
        self._dominant_modality = None


class ASGMLLoss(nn.Module):
    """
    ASGML Loss Function wrapper.

    This computes the multimodal loss and provides integration with
    the ASGML scheduler for determining update masks and staleness.

    Loss = L_fusion + γ * Σ_i (L_unimodal_i * 𝟙[update_i])

    The unimodal regularization terms are only applied when the
    corresponding modality updates.
    """

    def __init__(
        self,
        modalities: List[str],
        gamma: float = 1.0,  # Unimodal regularization weight (paper Eq. 12 default)
    ):
        """
        Args:
            modalities: List of modality names
            gamma: Weight for unimodal regularization terms
        """
        super().__init__()
        self.modalities = modalities
        self.gamma = gamma

    def forward(
        self,
        fusion_logits: torch.Tensor,
        unimodal_logits: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        update_mask: Optional[Dict[str, bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ASGML loss.

        Args:
            fusion_logits: Fused prediction logits (B, num_classes)
            unimodal_logits: Dict of unimodal logits per modality
            targets: Ground truth labels (B,)
            update_mask: Which modalities are updating (for conditional reg)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dict with individual loss components
        """
        # Fusion loss (always computed)
        fusion_loss = F.cross_entropy(fusion_logits, targets)

        # Unimodal losses
        unimodal_losses = {}
        for m in self.modalities:
            unimodal_losses[m] = F.cross_entropy(unimodal_logits[m], targets)

        # Compute regularization (conditional on update mask)
        if update_mask is None:
            update_mask = {m: True for m in self.modalities}

        unimodal_reg = torch.tensor(0.0, device=fusion_logits.device)
        for m in self.modalities:
            if update_mask[m]:
                unimodal_reg = unimodal_reg + unimodal_losses[m]

        # Total loss
        total_loss = fusion_loss + self.gamma * unimodal_reg

        # Collect all losses for logging
        loss_dict = {
            'fusion': fusion_loss,
            'unimodal_reg': unimodal_reg,
            'total': total_loss,
        }
        for m in self.modalities:
            loss_dict[f'unimodal_{m}'] = unimodal_losses[m]

        return total_loss, loss_dict


def compute_gradient_norms(
    model: nn.Module,
    modalities: List[str],
    encoder_prefix: str = "encoders",
) -> Dict[str, float]:
    """
    Compute gradient L2 norms for each modality encoder.

    Args:
        model: The multimodal model
        modalities: List of modality names
        encoder_prefix: Prefix for encoder parameters in model

    Returns:
        Dict mapping modality names to gradient norms
    """
    grad_norms = {}

    for modality in modalities:
        total_norm = 0.0
        count = 0

        for name, param in model.named_parameters():
            if f"{encoder_prefix}.{modality}" in name and param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
                count += 1

        grad_norms[modality] = (total_norm ** 0.5) if count > 0 else 0.0

    return grad_norms


def apply_staleness_gradients(
    model: nn.Module,
    staleness_buffer: StalenessBuffer,
    modality: str,
    staleness: int,
    scale: float = 1.0,
    encoder_prefix: str = "encoders",
) -> bool:
    """
    Replace current gradients with stale gradients from buffer.

    CRITICAL: This modifies gradients in-place before optimizer.step()

    Args:
        model: The multimodal model
        staleness_buffer: Buffer containing stored gradients
        modality: Which modality to apply stale gradients to
        staleness: How many steps old the gradients should be
        scale: Gradient scaling factor for compensation
        encoder_prefix: Prefix for encoder parameters

    Returns:
        True if stale gradients were applied, False otherwise
    """
    if staleness <= 0:
        return False

    stale_grads = staleness_buffer.get_stale_gradients(modality, staleness)
    if stale_grads is None:
        return False

    # Apply stale gradients to encoder parameters
    for name, param in model.named_parameters():
        if f"{encoder_prefix}.{modality}" in name:
            if name in stale_grads and param.grad is not None:
                param.grad.data = stale_grads[name].to(param.device) * scale

    return True
