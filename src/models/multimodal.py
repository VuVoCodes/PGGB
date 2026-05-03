"""
Multimodal model combining encoders, fusion, and classifier.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .encoders import AudioEncoder, VisualEncoder, TextEncoder, MLPEncoder
from .fusion import ConcatFusion, GatedFusion, SumFusion


class MultimodalModel(nn.Module):
    """
    General multimodal model supporting N modalities.
    """

    def __init__(
        self,
        modalities: List[str],
        num_classes: int,
        encoder_config: Dict,
        fusion_type: str = "concat",
        feature_dim: int = 512,
        fusion_dim: int = 512,
    ):
        """
        Args:
            modalities: List of modality names (e.g., ['audio', 'visual'])
            num_classes: Number of output classes
            encoder_config: Configuration dict for each encoder
            fusion_type: Type of fusion ('concat', 'gated', 'sum')
            feature_dim: Feature dimension for each encoder
            fusion_dim: Dimension after fusion
        """
        super().__init__()
        self.modalities = modalities
        self.num_modalities = len(modalities)
        self.feature_dim = feature_dim

        # Create encoders
        self.encoders = nn.ModuleDict()
        for modality in modalities:
            self.encoders[modality] = self._create_encoder(
                modality, encoder_config.get(modality, {}), feature_dim
            )

        # Create fusion module
        feature_dims = [feature_dim] * len(modalities)
        if fusion_type == "concat":
            self.fusion = ConcatFusion(feature_dims, fusion_dim)
        elif fusion_type == "gated":
            self.fusion = GatedFusion(feature_dims, fusion_dim)
        elif fusion_type == "sum":
            self.fusion = SumFusion(feature_dims, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # Unimodal classifiers (for regularization)
        self.unimodal_classifiers = nn.ModuleDict({
            modality: nn.Linear(feature_dim, num_classes)
            for modality in modalities
        })

    def _create_encoder(
        self, modality: str, config: Dict, feature_dim: int
    ) -> nn.Module:
        """Create encoder for a specific modality."""
        backbone = config.get("backbone", "resnet18")

        # MLP encoder for pre-extracted features (e.g., MOSEI)
        if backbone == "mlp":
            return MLPEncoder(
                input_dim=config.get("input_dim", 300),
                feature_dim=feature_dim,
                dropout=config.get("dropout", 0.3),
            )

        if modality == "audio":
            return AudioEncoder(
                backbone=backbone,
                pretrained=config.get("pretrained", True),
                feature_dim=feature_dim,
            )
        elif modality == "visual":
            return VisualEncoder(
                backbone=backbone,
                pretrained=config.get("pretrained", True),
                feature_dim=feature_dim,
            )
        elif modality == "text":
            return TextEncoder(
                vocab_size=config.get("vocab_size", 10000),
                embed_dim=config.get("embed_dim", 300),
                hidden_dim=config.get("hidden_dim", 256),
                feature_dim=feature_dim,
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the multimodal model.

        Args:
            inputs: Dictionary mapping modality names to input tensors
            return_features: Whether to return intermediate features

        Returns:
            logits: Fused prediction logits (B, num_classes)
            unimodal_logits: Dict of unimodal prediction logits (if return_features)
            features: Dict of encoder features (if return_features)
        """
        # Encode each modality
        features = {}
        for modality in self.modalities:
            features[modality] = self.encoders[modality](inputs[modality])

        # Fuse features
        feature_list = [features[m] for m in self.modalities]
        fused = self.fusion(feature_list)

        # Classify
        logits = self.classifier(fused)

        if return_features:
            # Compute unimodal predictions
            unimodal_logits = {
                modality: self.unimodal_classifiers[modality](features[modality])
                for modality in self.modalities
            }
            return logits, unimodal_logits, features

        return logits, None, None

    def get_encoder_parameters(self, modality: str):
        """Get parameters for a specific modality encoder."""
        return self.encoders[modality].parameters()

    def get_fusion_parameters(self):
        """Get parameters for fusion module and classifier."""
        return list(self.fusion.parameters()) + list(self.classifier.parameters())
