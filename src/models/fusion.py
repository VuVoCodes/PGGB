"""
Fusion modules for multimodal learning.
"""

import torch
import torch.nn as nn
from typing import List


class ConcatFusion(nn.Module):
    """Simple concatenation fusion (late fusion)."""

    def __init__(self, feature_dims: List[int], output_dim: int):
        """
        Args:
            feature_dims: List of feature dimensions for each modality
            output_dim: Output dimension after fusion
        """
        super().__init__()
        total_dim = sum(feature_dims)
        self.fc = nn.Linear(total_dim, output_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors, each of shape (B, feature_dim_i)
        Returns:
            Fused feature tensor of shape (B, output_dim)
        """
        concatenated = torch.cat(features, dim=1)
        return self.fc(concatenated)


class GatedFusion(nn.Module):
    """Gated fusion mechanism."""

    def __init__(self, feature_dims: List[int], output_dim: int):
        """
        Args:
            feature_dims: List of feature dimensions for each modality
            output_dim: Output dimension after fusion
        """
        super().__init__()
        self.n_modalities = len(feature_dims)

        # Project each modality to common dimension
        self.projectors = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])

        # Gate networks for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.Sigmoid()
            ) for dim in feature_dims
        ])

        # Final projection
        self.output_fc = nn.Linear(output_dim, output_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors, each of shape (B, feature_dim_i)
        Returns:
            Fused feature tensor of shape (B, output_dim)
        """
        # Project and gate each modality
        gated_features = []
        for i, feat in enumerate(features):
            projected = self.projectors[i](feat)
            gate = self.gates[i](feat)
            gated_features.append(projected * gate)

        # Sum gated features
        fused = sum(gated_features)
        return self.output_fc(fused)


class SumFusion(nn.Module):
    """Simple sum fusion after projection to common dimension."""

    def __init__(self, feature_dims: List[int], output_dim: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(feat) for proj, feat in zip(self.projectors, features)]
        return sum(projected)
