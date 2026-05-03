"""
Modality-specific encoders for multimodal learning.

OGM-GE compatible: Trains from scratch with kaiming/xavier initialization.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def ogm_ge_weight_init(m):
    """
    OGM-GE weight initialization (matches their utils.py weight_init).

    - Conv2d: kaiming_normal (fan_out, relu)
    - Linear: xavier_normal, bias=0
    - BatchNorm2d: weight=1, bias=0
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class VisualEncoder(nn.Module):
    """Visual encoder using ResNet backbone (OGM-GE compatible with temporal input)."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,  # OGM-GE default: train from scratch
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load backbone (OGM-GE trains from scratch by default)
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.in_features = in_features

        # Project to feature_dim if needed
        if in_features != feature_dim:
            self.projector = nn.Linear(in_features, feature_dim)
        else:
            self.projector = nn.Identity()

        # Apply OGM-GE weight initialization if not using pretrained
        if not pretrained:
            self.apply(ogm_ge_weight_init)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone layers without avgpool/fc flatten."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x  # (B, C, H, W) spatial features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass matching OGM-GE visual encoder.

        Args:
            x: Visual input tensor of shape (B, C, T, H, W) or (B, C, H, W)
               Where T is the number of frames (temporal dimension)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Handle both 4D (B, C, H, W) and 5D (B, C, T, H, W) inputs
        if x.dim() == 5:
            # OGM-GE temporal handling: reshape (B, C, T, H, W) -> (B*T, C, H, W)
            B, C, T, H, W = x.size()
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
            x = x.view(B * T, C, H, W)  # (B*T, C, H, W)

            # Process through ResNet layers (without avgpool/fc)
            features = self._forward_backbone(x)  # (B*T, C', H', W')

            # Reshape back and pool over time
            _, C_out, H_out, W_out = features.size()
            features = features.view(B, T, C_out, H_out, W_out)  # (B, T, C', H', W')
            features = features.permute(0, 2, 1, 3, 4)  # (B, C', T, H', W')

            # 3D adaptive average pooling (OGM-GE style)
            features = nn.functional.adaptive_avg_pool3d(features, 1)  # (B, C', 1, 1, 1)
            features = torch.flatten(features, 1)  # (B, C')
        else:
            # Standard 4D input: (B, C, H, W)
            features = self._forward_backbone(x)  # (B, C', H', W')
            features = nn.functional.adaptive_avg_pool2d(features, 1)  # (B, C', 1, 1)
            features = torch.flatten(features, 1)  # (B, C')

        return self.projector(features)


class AudioEncoder(nn.Module):
    """Audio encoder for spectrogram inputs using ResNet backbone (OGM-GE compatible)."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,  # OGM-GE default: train from scratch
        feature_dim: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Load backbone (OGM-GE trains from scratch)
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Modify first conv layer for single-channel spectrogram input (OGM-GE exact)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove classification head
        self.backbone.fc = nn.Identity()

        # Project to feature_dim if needed
        if in_features != feature_dim:
            self.projector = nn.Linear(in_features, feature_dim)
        else:
            self.projector = nn.Identity()

        # Apply OGM-GE weight initialization (always, since audio starts from scratch)
        self.apply(ogm_ge_weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio spectrogram tensor of shape (B, 1, H, W)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        features = self.backbone(x)
        return self.projector(features)


class TextEncoder(nn.Module):
    """Text encoder using simple embedding + LSTM."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        feature_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.projector = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape (B, seq_len)
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.projector(hidden)


class MLPEncoder(nn.Module):
    """MLP encoder for pre-extracted features (e.g., CMU-MOSEI).

    Matches InfoReg's TriModalClassifier encoder architecture:
    Linear → ReLU → Dropout → Linear → ReLU
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Pre-extracted features of shape (B, D) or (B, T, D).
               If 3D, mean-pools over the temporal dimension first.
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)
