"""
Twitter15 dataset (Yu & Jiang, IJCAI 2019 / TomBERT).

Aspect-based sentiment analysis on text+image tweets. Loads pre-extracted
BERT (text, 768-d) and ResNet18 (image, 512-d) features from
`data/Twitter15/features/{train,valid,test}.pt`. Run
`scripts/extract_text_image_features.py --dataset twitter` to generate.

Splits (matching AUG paper, NeurIPS 2025):
- Train: 3,179
- Valid: 1,122
- Test: 1,037

Labels: 3 classes (negative=0, neutral=1, positive=2).
"""

import os
from typing import Dict

import torch
from torch.utils.data import Dataset


class TwitterDataset(Dataset):
    """Twitter15 dataset with pre-extracted text+image features.

    Returns dict with keys {"text", "image", "label"} for each sample.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        feat_path = os.path.join(root, "features", f"{split}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Pre-extracted features not found: {feat_path}\n"
                f"Run: python scripts/extract_text_image_features.py --dataset twitter"
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
