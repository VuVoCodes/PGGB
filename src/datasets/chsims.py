"""
CH-SIMS Dataset for multimodal sentiment analysis (3 modalities).

CH-SIMS contains Mandarin Chinese video segments with pre-extracted
features for text, audio, and vision, distributed through the MMSA toolkit
as ``unaligned_39.pkl``:

    text   BERT (Chinese BERT-base) token features, 39 x 768
    audio  LibROSA acoustic features, 400 x 33
    vision OpenFace 2.0 facial features, 55 x 709
    labels 3-class sentiment (0 = negative, 1 = neutral, 2 = positive)

Splits are 1,368 train / 456 valid / 457 test.

Reference:
Yu et al., "CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with
Fine-grained Annotation of Modality" (ACL 2020)
"""

import os
import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class CHSIMSDataset(Dataset):
    """CH-SIMS Dataset with pre-extracted MMSA features.

    Parameters
    ----------
    root : str
        Root directory containing the pickle file.
    split : str
        'train', 'valid', or 'test'.
    pkl_path : str or None
        Explicit path to pickle file. If None, auto-detects from root.
    check_provenance : bool
        If True (default), verify that the loaded split matches the expected
        CH-SIMS sample count and feature dimensions, and raise otherwise.
        This guards against silently training on a different MMSA pickle,
        e.g. the similarly named CMU-MOSEI file ``unaligned_50.pkl``.
    """

    # Known pickle locations to search (in priority order)
    _PICKLE_CANDIDATES = [
        "Processed/Processed/unaligned_39.pkl",    # MMSA layout
        "unaligned_39.pkl",                        # MMSA (flat)
    ]

    # Expected CH-SIMS provenance: split -> sample count
    _EXPECTED_SIZES = {"train": 1368, "valid": 456, "test": 457}
    # Expected feature dimensions: text, audio, vision
    _EXPECTED_DIMS = (768, 33, 709)

    def __init__(
        self,
        root: str,
        split: str = "train",
        pkl_path: str = None,
        check_provenance: bool = True,
    ):
        super().__init__()

        # Find pickle file
        if pkl_path is not None:
            dataset_path = pkl_path
        else:
            dataset_path = self._find_pickle(root)

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"CH-SIMS pickle not found. Searched in {root}.\n"
                f"Expected one of: {self._PICKLE_CANDIDATES}\n"
                f"Run scripts/prepare_chsims.py for download instructions."
            )

        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        split_data = dataset[split]
        self._validate(split_data, split, dataset_path, check_provenance)

        # Load features
        vision = split_data["vision"].astype(np.float32)
        text = split_data["text"].astype(np.float32)

        # Handle -inf in audio features
        audio = split_data["audio"].astype(np.float32)
        audio[audio == -np.inf] = 0
        audio[np.isnan(audio)] = 0

        # Z-score normalize per feature dimension to prevent training instability
        # (MMSA features have extreme value ranges, e.g. audio min=-827).
        # Statistics are computed per split, over all (padded) time steps.
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

        # MMSA classification labels are already 3-class (0, 1, 2)
        self._classification_labels = torch.tensor(
            split_data["classification_labels"].astype(np.int64)
        ).cpu().detach()

        self.n_modalities = 3

        # Report stats
        print(
            f"CH-SIMS [{split}]: {len(self)} samples | "
            f"text={list(self.text.shape)}, "
            f"audio={list(self.audio.shape)}, "
            f"vision={list(self.vision.shape)}"
        )

    @classmethod
    def _validate(
        cls,
        split_data: dict,
        split: str,
        dataset_path: str,
        check_provenance: bool,
    ) -> None:
        """Check that the pickle really is the CH-SIMS MMSA file."""
        if "classification_labels" not in split_data:
            raise ValueError(
                f"{dataset_path} is not an MMSA CH-SIMS pickle: split "
                f"'{split}' has no 'classification_labels' key. Keys: "
                f"{list(split_data.keys())}"
            )
        if not check_provenance:
            return

        n = split_data["text"].shape[0]
        dims = (
            split_data["text"].shape[-1],
            split_data["audio"].shape[-1],
            split_data["vision"].shape[-1],
        )
        expected_n = cls._EXPECTED_SIZES.get(split)
        if (expected_n is not None and n != expected_n) or dims != cls._EXPECTED_DIMS:
            raise ValueError(
                f"{dataset_path} does not look like CH-SIMS. Split '{split}' has "
                f"{n} samples with feature dims (text, audio, vision) = {dims}, "
                f"expected {expected_n} samples and {cls._EXPECTED_DIMS}. "
                f"CMU-MOSEI's MMSA file is 'unaligned_50.pkl' and is a different "
                f"dataset. Pass check_provenance=False to bypass this check."
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
        return {
            "text": self.text[index],
            "audio": self.audio[index],
            "vision": self.vision[index],
            "label": self._classification_labels[index].long(),
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
