"""
UPMC-Food101 Dataset (Wang et al., ICME 2015).

Multimodal food classification: image + recipe title text → 101-class label.

Loads pre-extracted frozen BERT (768-d) + frozen ResNet18 (512-d) features from
`data/Food101/features/{train,test}.pt`. Run
`scripts/extract_text_image_features.py --dataset food101` to generate.

Splits (standard UPMC-Food101):
- Train: ~67K
- Test: ~23K
- Classes: 101

Known large modality imbalance: text dominates image by ~16-20pp (per CGGM paper).
"""

import os
from typing import Dict

import torch
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    """UPMC-Food101 with pre-extracted text+image features."""

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        feat_path = os.path.join(root, "features", f"{split}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Pre-extracted features not found: {feat_path}\n"
                f"Run: python scripts/extract_text_image_features.py --dataset food101"
            )
        data = torch.load(feat_path, weights_only=True)
        self.text = data["text"].float()      # (N, 768)
        self.image = data["image"].float()    # (N, 512)
        self.label = data["label"].long()     # (N,)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "text": self.text[idx],
            "image": self.image[idx],
            "label": self.label[idx],
        }

    @property
    def text_dim(self):
        return self.text.shape[-1]

    @property
    def image_dim(self):
        return self.image.shape[-1]
