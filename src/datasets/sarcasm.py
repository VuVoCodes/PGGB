"""
Multi-modal Sarcasm Detection dataset (Cai et al., ACL 2019).

Loads pre-extracted BERT (text, 768-d) and ResNet18 (image, 512-d) features
from `data/Sarcasm/features/{train,valid,test}.pt`. Run
`scripts/extract_text_image_features.py --dataset sarcasm` to generate.

Splits (matching AUG paper, NeurIPS 2025):
- Train: 19,816
- Valid: 2,410
- Test: 2,409
"""

import os
from typing import Dict

import torch
from torch.utils.data import Dataset


class SarcasmDataset(Dataset):
    """Sarcasm detection dataset with pre-extracted text+image features.

    Returns dict with keys {"text", "image", "label"} for each sample.
    """

    def __init__(self, root: str, split: str = "train"):
        super().__init__()
        feat_path = os.path.join(root, "features", f"{split}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Pre-extracted features not found: {feat_path}\n"
                f"Run: python scripts/extract_text_image_features.py --dataset sarcasm"
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
