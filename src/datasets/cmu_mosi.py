"""
CMU-MOSI Dataset for multimodal sentiment analysis.

Pre-extracted features: text (GloVe 300d), audio (COVAREP 74d), vision (FACET 35d).
Sentiment labels: continuous [-3, +3], binarized for classification.

Reference:
Zadeh et al., "Multimodal Sentiment Intensity Analysis in Videos"
AAAI 2017
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


class CMUMOSIDataset(Dataset):
    """CMU-MOSI dataset with pre-extracted features.

    Expected data format: pickle file with structure:
    {
        'train': {'text': (N, T, 300), 'audio': (N, T, 74), 'vision': (N, T, 35), 'labels': (N, 1, 1)},
        'valid': {...},
        'test': {...}
    }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        binary: bool = True,
    ):
        """
        Args:
            root: Path to directory containing mosi_raw.pkl
            split: 'train', 'valid', or 'test'
            binary: If True, binarize labels (positive/negative sentiment)
        """
        self.root = root
        self.split = split
        self.binary = binary

        # Find pickle file
        pkl_path = None
        for name in ['mosi_raw.pkl', 'mosi_data.pkl', 'mosi.pkl']:
            candidate = os.path.join(root, name)
            if os.path.exists(candidate):
                pkl_path = candidate
                break

        if pkl_path is None:
            raise FileNotFoundError(f"No MOSI pickle found in {root}")

        dataset = pickle.load(open(pkl_path, 'rb'))

        self.text = torch.tensor(dataset[split]['text'].astype(np.float32))
        self.audio = dataset[split]['audio'].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio)
        self.vision = torch.tensor(dataset[split]['vision'].astype(np.float32))
        self.labels_raw = dataset[split]['labels'].astype(np.float32).squeeze()

        if binary:
            # Binarize: positive (>0) = 1, negative (<=0) = 0
            self.labels = torch.tensor((self.labels_raw > 0).astype(np.int64))
        else:
            self.labels = torch.tensor(self.labels_raw)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Pool over time dimension (mean pooling) to get fixed-size features
        text = self.text[idx].mean(dim=0)      # (300,)
        audio = self.audio[idx].mean(dim=0)    # (74,)
        vision = self.vision[idx].mean(dim=0)  # (35,)

        return {
            "text": text,
            "audio": audio,
            "visual": vision,
            "label": self.labels[idx],
        }

    @property
    def num_classes(self) -> int:
        return 2 if self.binary else 1

    def get_dims(self) -> Dict[str, int]:
        return {"text": 300, "audio": 74, "visual": 35}
