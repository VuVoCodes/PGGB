"""
Probe networks for independent modality monitoring.

CRITICAL SAFETY RULE: Probes must NEVER backpropagate into encoders.
- Always use .detach() on encoder features before passing to probes
- Probes have their own separate optimizers
- Probe training is completely decoupled from main model training

Probes measure modality utilization - how discriminative each modality's
representations are independently of the fusion head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LinearProbe(nn.Module):
    """
    Simple linear probe for monitoring modality utilization.

    Linear probes are sufficient for detecting suppression (Alain & Bengio, 2016)
    and have minimal computational overhead.
    """

    def __init__(self, feature_dim: int, num_classes: int):
        """
        Args:
            feature_dim: Dimension of input features from encoder
            num_classes: Number of output classes
        """
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: MUST be detached from encoder computation graph
                     Shape: (B, feature_dim)
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.classifier(features)


class MLPProbe(nn.Module):
    """
    Single hidden layer MLP probe for slightly more expressive monitoring.

    Use when linear probe shows too much variance or when features
    require some nonlinear transformation to be discriminative.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: Dimension of input features from encoder
            num_classes: Number of output classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: MUST be detached from encoder computation graph
                     Shape: (B, feature_dim)
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.net(features)


class ProbeManager:
    """
    Manages probe networks for all modalities with safety guarantees.

    This class ensures:
    1. Probes never backpropagate into encoders (via explicit .detach())
    2. Probes have completely separate optimizers
    3. Probe training is decoupled from main training loop
    4. Utilization metrics are computed correctly
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int,
        num_classes: int,
        probe_type: str = "linear",
        probe_lr: float = 1e-3,
        device: torch.device = None,
        ema_alpha: float = 0.1,
    ):
        """
        Args:
            modalities: List of modality names
            feature_dim: Feature dimension from encoders
            num_classes: Number of output classes
            probe_type: 'linear' or 'mlp_1layer'
            probe_lr: Learning rate for probe optimizers
            device: Device to place probes on
            ema_alpha: EMA smoothing factor for probe accuracies (0.1 = slow, 0.9 = fast)
        """
        self.modalities = modalities
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.probe_type = probe_type
        self.device = device or torch.device('cpu')
        self.ema_alpha = ema_alpha

        # Create probes for each modality
        self.probes: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}

        for modality in modalities:
            probe = self._create_probe()
            probe = probe.to(self.device)
            self.probes[modality] = probe
            self.optimizers[modality] = torch.optim.Adam(
                probe.parameters(), lr=probe_lr
            )

        # Track probe accuracies for utilization measurement
        self.accuracy_history: Dict[str, List[float]] = {
            m: [] for m in modalities
        }

        # EMA-smoothed accuracies for stable adaptive scheduling
        # Raw per-batch probe accuracy has high variance (batch_size=64, 6 classes)
        # EMA smoothing prevents erratic adaptation decisions
        self.accuracy_ema: Dict[str, float] = {m: 0.0 for m in modalities}

    def _create_probe(self) -> nn.Module:
        """Create a probe network based on probe_type."""
        if self.probe_type == "linear":
            return LinearProbe(self.feature_dim, self.num_classes)
        elif self.probe_type == "mlp_1layer":
            return MLPProbe(self.feature_dim, self.num_classes)
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")

    def train_probes(
        self,
        features: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        num_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Train probes on detached features.

        SAFETY: This method explicitly detaches features to ensure
        no gradients flow back to encoders.

        Args:
            features: Dict mapping modality names to feature tensors
                     These will be DETACHED inside this method
            targets: Ground truth labels (B,)
            num_steps: Number of optimization steps per call

        Returns:
            Dict mapping modality names to training losses
        """
        losses = {}

        for modality in self.modalities:
            probe = self.probes[modality]
            optimizer = self.optimizers[modality]

            # CRITICAL: Detach features to prevent backprop into encoder
            # Convert to float32 for probe training (handles AMP float16 features)
            feat = features[modality].detach().float()

            probe.train()
            for _ in range(num_steps):
                optimizer.zero_grad()
                logits = probe(feat)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()

            losses[modality] = loss.item()

        return losses

    @torch.no_grad()
    def evaluate_probes(
        self,
        features: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate probe accuracy on given features.

        SAFETY: Uses torch.no_grad() and explicit detach for safety.

        Args:
            features: Dict mapping modality names to feature tensors
            targets: Ground truth labels (B,)

        Returns:
            Dict with 'accuracy' and 'loss' for each modality
        """
        results = {}

        for modality in self.modalities:
            probe = self.probes[modality]
            probe.eval()

            # CRITICAL: Detach features
            # Convert to float32 for probe evaluation (handles AMP float16 features)
            feat = features[modality].detach().float()

            logits = probe(feat)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean().item()

            results[modality] = {
                'accuracy': accuracy,
                'loss': loss.item(),
            }

            # Track history
            self.accuracy_history[modality].append(accuracy)

            # Update EMA-smoothed accuracy
            # EMA = α * new_value + (1-α) * old_ema
            old_ema = self.accuracy_ema[modality]
            self.accuracy_ema[modality] = (
                self.ema_alpha * accuracy + (1 - self.ema_alpha) * old_ema
            )

        return results

    def compute_utilization_gap(self, use_ema: bool = True) -> Optional[float]:
        """
        Compute utilization gap between modalities.

        Utilization gap = max(probe_acc) - min(probe_acc)

        A large gap indicates modality imbalance - one modality's
        representations are much more discriminative than another's.

        Args:
            use_ema: If True, use EMA-smoothed accuracies (more stable).

        Returns:
            Utilization gap, or None if not enough history
        """
        if not all(len(h) > 0 for h in self.accuracy_history.values()):
            return None

        if use_ema:
            accs = list(self.accuracy_ema.values())
        else:
            # Use most recent raw accuracies
            accs = [h[-1] for h in self.accuracy_history.values()]

        return max(accs) - min(accs)

    def get_dominant_modality(self) -> Optional[str]:
        """
        Identify which modality is currently dominant (highest probe accuracy).

        Returns:
            Name of dominant modality, or None if not enough history
        """
        if not all(len(h) > 0 for h in self.accuracy_history.values()):
            return None

        recent_accs = {m: h[-1] for m, h in self.accuracy_history.items()}
        return max(recent_accs, key=recent_accs.get)

    def get_weak_modality(self) -> Optional[str]:
        """
        Identify which modality is currently weakest (lowest probe accuracy).

        Returns:
            Name of weakest modality, or None if not enough history
        """
        if not all(len(h) > 0 for h in self.accuracy_history.values()):
            return None

        recent_accs = {m: h[-1] for m, h in self.accuracy_history.items()}
        return min(recent_accs, key=recent_accs.get)

    def get_utilization_scores(self, use_ema: bool = True) -> Dict[str, float]:
        """
        Get current utilization score (probe accuracy) for each modality.

        Args:
            use_ema: If True, return EMA-smoothed accuracies (more stable for adaptive control).
                     If False, return raw most-recent accuracies.

        Returns:
            Dict mapping modality names to their utilization scores
        """
        if use_ema:
            # Return EMA-smoothed values for stable adaptive scheduling
            return self.accuracy_ema.copy()
        else:
            # Return raw most-recent values
            scores = {}
            for modality in self.modalities:
                if self.accuracy_history[modality]:
                    scores[modality] = self.accuracy_history[modality][-1]
                else:
                    scores[modality] = 0.0
            return scores

    def reset_history(self) -> None:
        """Reset accuracy history and EMA (call at start of training or new phase)."""
        self.accuracy_history = {m: [] for m in self.modalities}
        self.accuracy_ema = {m: 0.0 for m in self.modalities}

    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            'probes': {m: p.state_dict() for m, p in self.probes.items()},
            'optimizers': {m: o.state_dict() for m, o in self.optimizers.items()},
            'accuracy_history': self.accuracy_history,
            'accuracy_ema': self.accuracy_ema,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict from checkpoint."""
        for m in self.modalities:
            self.probes[m].load_state_dict(state_dict['probes'][m])
            self.optimizers[m].load_state_dict(state_dict['optimizers'][m])
        self.accuracy_history = state_dict['accuracy_history']
        if 'accuracy_ema' in state_dict:
            self.accuracy_ema = state_dict['accuracy_ema']
