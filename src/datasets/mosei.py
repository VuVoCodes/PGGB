"""
CMU-MOSEI Dataset for multimodal sentiment analysis (3 modalities).

CMU-MOSEI contains utterances from YouTube opinion videos,
with pre-extracted features for text, audio, and vision.

Supports two pickle formats:
1. InfoReg format (mosei_senti_data.pkl): text(300d), audio(74d), vision(35d)
2. MMSA format (unaligned_39.pkl): text(768d), audio(33d), vision(709d)

Reference:
Zadeh et al., "Multimodal Language Analysis in the Wild:
CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph" (ACL 2018)
"""

import os
import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MOSEIDataset(Dataset):
    """CMU-MOSEI Dataset with pre-extracted features.

    Parameters
    ----------
    root : str
        Root directory containing the pickle file.
    split : str
        'train', 'valid', or 'test'.
    pkl_path : str or None
        Explicit path to pickle file. If None, auto-detects from root.
    """

    # Known pickle locations to search (in priority order)
    _PICKLE_CANDIDATES = [
        "mosei_senti_data.pkl",                    # InfoReg format
        "Processed/Processed/unaligned_39.pkl",    # MMSA format
        "unaligned_39.pkl",                        # MMSA (flat)
        "Processed/Processed/aligned_50.pkl",      # MMSA aligned
        "aligned_50.pkl",                          # MMSA aligned (flat)
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        pkl_path: str = None,
    ):
        super().__init__()

        # Find pickle file
        if pkl_path is not None:
            dataset_path = pkl_path
        else:
            dataset_path = self._find_pickle(root)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"MOSEI pickle not found. Searched in {root}.\n"
                f"Expected one of: {self._PICKLE_CANDIDATES}\n"
                f"Run scripts/prepare_mosei.py to download."
            )

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        # Detect format and load accordingly
        split_data = dataset[split]
        self._format = self._detect_format(split_data)

        # Load features
        vision = split_data["vision"].astype(np.float32)
        text = split_data["text"].astype(np.float32)

        # Handle -inf in audio features (common in COVAREP)
        audio = split_data["audio"].astype(np.float32)
        audio[audio == -np.inf] = 0
        audio[np.isnan(audio)] = 0

        # Z-score normalize per feature dimension to prevent training instability
        # (MMSA features have extreme value ranges, e.g., audio min=-827)
        for arr in [text, audio, vision]:
            # Reshape to 2D for stats: (N*T, D)
            orig_shape = arr.shape
            flat = arr.reshape(-1, arr.shape[-1])
            mean = flat.mean(axis=0, keepdims=True)
            std = flat.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0  # Avoid division by zero
            flat[:] = (flat - mean) / std
            arr[:] = flat.reshape(orig_shape)

        self.vision = torch.tensor(vision).cpu().detach()
        self.text = torch.tensor(text).cpu().detach()
        self.audio = torch.tensor(audio).cpu().detach()

        # Load labels based on format
        if self._format == "mmsa":
            # MMSA format: classification_labels already 3-class (0, 1, 2)
            self._classification_labels = torch.tensor(
                split_data["classification_labels"].astype(np.int64)
            ).cpu().detach()
        else:
            # InfoReg format: continuous labels, convert to 3-class
            self._raw_labels = torch.tensor(
                split_data["labels"].astype(np.float32)
            ).cpu().detach()

        self.n_modalities = 3

        # Report stats
        print(
            f"MOSEI [{split}] ({self._format}): {len(self)} samples | "
            f"text={list(self.text.shape)}, "
            f"audio={list(self.audio.shape)}, "
            f"vision={list(self.vision.shape)}"
        )

    @staticmethod
    def _detect_format(split_data: dict) -> str:
        """Detect pickle format based on available keys."""
        if "classification_labels" in split_data:
            return "mmsa"
        elif "labels" in split_data:
            return "inforeg"
        else:
            raise ValueError(
                f"Unknown MOSEI pickle format. Keys: {list(split_data.keys())}"
            )

    def _find_pickle(self, root: str) -> str:
        """Search for pickle file in standard locations."""
        for candidate in self._PICKLE_CANDIDATES:
            path = os.path.join(root, candidate)
            if os.path.exists(path):
                return path
        # Return first candidate for error message
        return os.path.join(root, self._PICKLE_CANDIDATES[0])

    def __len__(self) -> int:
        return self.text.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self._format == "mmsa":
            target = self._classification_labels[index].long()
        else:
            # InfoReg: convert continuous to 3-class
            Y = self._raw_labels[index]
            threshold = 0.5
            if Y <= -threshold:
                target = torch.tensor(0, dtype=torch.long)
            elif Y >= threshold:
                target = torch.tensor(2, dtype=torch.long)
            else:
                target = torch.tensor(1, dtype=torch.long)

        return {
            "text": self.text[index],
            "audio": self.audio[index],
            "vision": self.vision[index],
            "label": target,
        }

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def text_dim(self) -> int:
        """Dimension of text features."""
        return self.text.shape[-1]

    @property
    def audio_dim(self) -> int:
        """Dimension of audio features."""
        return self.audio.shape[-1]

    @property
    def visual_dim(self) -> int:
        """Dimension of vision features."""
        return self.vision.shape[-1]
