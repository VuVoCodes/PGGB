"""
CGGM: Classifier-Guided Gradient Modulation (Guo et al., NeurIPS 2024)

Implements both gradient magnitude and direction modulation:
- Magnitude: scale encoder gradients inversely proportional to modality improvement
- Direction: align fusion gradient direction toward lagging modalities via L_gm loss

Reference: https://github.com/zrguo/CGGM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ModalityClassifier(nn.Module):
    """Simple linear classifier for a single modality's features.

    Matches CGGM's approach: takes detached encoder features, predicts class.
    Must have an `out_layer` attribute for gradient extraction.
    """

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is not None:
            self.layers = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.out_layer = nn.Linear(hidden_dim, num_classes)
        else:
            self.layers = nn.Identity()
            self.out_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return self.out_layer(x)


class CGGMModule:
    """CGGM gradient modulation manager.

    Manages per-modality classifiers and computes:
    1. Accuracy-based magnitude scaling coefficients
    2. Gradient direction alignment loss (L_gm)
    3. Encoder gradient scaling

    Parameters
    ----------
    modalities : list of str
        Modality names (e.g., ['audio', 'visual'])
    feature_dim : int
        Encoder output feature dimension
    num_classes : int
        Number of output classes
    rou : float
        Gradient scaling amplifier (default: 1.3)
    lamda : float
        Weight for gradient direction loss L_gm (default: 0.2)
    cls_lr : float
        Classifier learning rate (default: 5e-4)
    cls_hidden : int or None
        Hidden dim for classifiers (None = linear)
    device : str
        Device for classifiers
    """

    def __init__(
        self,
        modalities: List[str],
        feature_dim: int,
        num_classes: int,
        rou: float = 1.3,
        lamda: float = 0.2,
        cls_lr: float = 5e-4,
        cls_hidden: Optional[int] = None,
        device: str = "cuda",
    ):
        self.modalities = modalities
        self.num_mod = len(modalities)
        self.rou = rou
        self.lamda = lamda
        self.device = device
        self.num_classes = num_classes

        # Per-modality classifiers
        self.classifiers = nn.ModuleDict({
            m: ModalityClassifier(feature_dim, num_classes, cls_hidden)
            for m in modalities
        }).to(device)

        # Separate optimizer for classifiers
        self.cls_optimizer = torch.optim.Adam(
            self.classifiers.parameters(), lr=cls_lr
        )

        # Previous batch accuracy per modality (for Δε computation)
        self.prev_acc = {m: 0.0 for m in modalities}

        # Store L_gm from previous iteration (applied to next iteration's loss)
        self.l_gm = None

    def compute_classifier_accuracy(
        self,
        cls_preds: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-modality accuracy from classifier predictions."""
        acc = {}
        for m in self.modalities:
            preds = cls_preds[m]
            if self.num_classes == 1:
                # Regression: use negative MAE as "accuracy"
                acc[m] = -F.l1_loss(preds.squeeze(), labels.float()).item()
            else:
                # Classification: standard accuracy
                pred_labels = preds.argmax(dim=-1)
                correct = (pred_labels == labels).float().mean().item()
                acc[m] = correct
        return acc

    def compute_coefficients(
        self, current_acc: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute magnitude balancing coefficients B_m.

        B_m = (sum(Δε) - Δε_m) / sum(Δε)
        Modalities with lower improvement get higher coefficients.
        """
        diff = {m: current_acc[m] - self.prev_acc[m] for m in self.modalities}
        diff_sum = sum(diff.values()) + 1e-8

        coeff = {}
        for m in self.modalities:
            coeff[m] = (diff_sum - diff[m]) / diff_sum

        # Update previous accuracy
        self.prev_acc = current_acc

        return coeff

    def compute_l_gm(
        self,
        coeff: Dict[str, float],
        cls_grads: Dict[str, torch.Tensor],
        fusion_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient direction loss L_gm as a differentiable tensor.

        L_gm = (1/M) * sum_m (|B_m| - B_m * cos_sim(fusion_grad, cls_grad_m))
        """
        fusion_flat = fusion_grad.view(-1)
        l_gm = torch.tensor(0.0, device=fusion_grad.device)

        for m in self.modalities:
            cls_flat = cls_grads[m].view(-1)
            cos_sim = F.cosine_similarity(cls_flat, fusion_flat, dim=0)
            l_gm = l_gm + abs(coeff[m]) - coeff[m] * cos_sim

        l_gm = l_gm / self.num_mod
        return l_gm

    def scale_encoder_gradients(
        self,
        model: nn.Module,
        coeff: Dict[str, float],
    ):
        """Scale encoder gradients by B_m * rou."""
        for m in self.modalities:
            scale = coeff[m] * self.rou
            for name, param in model.named_parameters():
                if f"encoders.{m}" in name and param.grad is not None:
                    param.grad *= scale

    def step(
        self,
        model: nn.Module,
        features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, Dict[str, float]]:
        """Full CGGM step after main loss backward.

        Call this AFTER main_loss.backward() but BEFORE optimizer.step().

        Returns:
            l_gm_value: The L_gm loss value for logging
            coeff: The magnitude scaling coefficients
        """
        # 1. Forward classifiers on detached features
        self.cls_optimizer.zero_grad()
        cls_preds = {}
        for m in self.modalities:
            feat = features[m].detach()
            cls_preds[m] = self.classifiers[m](feat)

        # 2. Compute classifier loss and backward
        cls_loss = sum(
            criterion(cls_preds[m], labels) for m in self.modalities
        )
        cls_loss.backward()

        # 3. Extract classifier output layer gradients
        cls_grads = {}
        for m in self.modalities:
            cls_grads[m] = self.classifiers[m].out_layer.weight.grad

        # 4. Extract fusion classifier gradient (from main model)
        fusion_grad = None
        for name, param in model.named_parameters():
            if "classifier.weight" in name and param.grad is not None:
                fusion_grad = param.grad
                break

        # Fallback: try unimodal_classifiers or other naming
        if fusion_grad is None:
            for name, param in model.named_parameters():
                if "classifier" in name and "weight" in name and param.grad is not None:
                    fusion_grad = param.grad
                    break

        # 5. Compute per-modality accuracy and coefficients
        current_acc = self.compute_classifier_accuracy(cls_preds, labels)
        coeff = self.compute_coefficients(current_acc)

        # 6. Compute L_gm (gradient direction loss)
        l_gm_value = 0.0
        if fusion_grad is not None:
            l_gm_tensor = self.compute_l_gm(coeff, cls_grads, fusion_grad)
            l_gm_value = l_gm_tensor.item()
            self.l_gm = l_gm_value
        else:
            self.l_gm = None

        # 7. Scale encoder gradients by coefficients
        self.scale_encoder_gradients(model, coeff)

        # 8. Step classifier optimizer
        self.cls_optimizer.step()

        return l_gm_value, coeff

    def get_l_gm_for_loss(self) -> Optional[float]:
        """Get L_gm scalar from previous iteration to add to current loss.

        Returns None if no L_gm has been computed yet, otherwise a float
        that should be multiplied by lamda and added to the main loss.
        Note: L_gm should be small (typically 0-2 range). If it grows
        beyond 10, something is wrong with the coefficient computation.
        """
        if self.l_gm is not None and abs(self.l_gm) > 100:
            # Clamp to prevent loss explosion
            return max(min(self.l_gm, 10.0), -10.0)
        return self.l_gm
