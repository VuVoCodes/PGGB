#!/usr/bin/env python3
"""
Prepare the CH-SIMS dataset for training.

CH-SIMS is distributed through the MMSA toolkit as a pre-extracted feature
pickle (``unaligned_39.pkl``) containing:
  - text:   BERT (Chinese BERT-base) token features, 39 x 768
  - audio:  LibROSA acoustic features, 400 x 33
  - vision: OpenFace 2.0 facial features, 55 x 709
  - classification_labels: 3-class sentiment (0 = neg, 1 = neu, 2 = pos)

Splits are 1,368 train / 456 valid / 457 test.

The file is not downloadable without accepting the dataset terms, so this
script only locates and verifies it.

Reference:
Yu et al., "CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with
Fine-grained Annotation of Modality" (ACL 2020)

Usage:
    python scripts/prepare_chsims.py [--data-root data/CH-SIMS]
"""

import argparse
import hashlib
import os
import pickle
import sys

EXPECTED_SIZES = {"train": 1368, "valid": 456, "test": 457}
EXPECTED_DIMS = {"text": 768, "audio": 33, "vision": 709}

CANDIDATES = [
    "Processed/Processed/unaligned_39.pkl",
    "unaligned_39.pkl",
]


def find_pickle(data_root: str) -> str:
    """Return the first existing candidate path, else the preferred one."""
    for candidate in CANDIDATES:
        path = os.path.join(data_root, candidate)
        if os.path.exists(path):
            return path
    return os.path.join(data_root, CANDIDATES[0])


def verify_pickle(pkl_path: str) -> bool:
    """Verify the pickle really is CH-SIMS and not another MMSA dataset.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file.

    Returns
    -------
    bool
        True if the file matches the expected CH-SIMS structure.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR loading pickle: {exc}")
        return False

    ok = True
    for split, expected_n in EXPECTED_SIZES.items():
        if split not in data:
            print(f"  ERROR: missing split '{split}'")
            ok = False
            continue

        split_data = data[split]
        for key in ["text", "audio", "vision", "classification_labels"]:
            if key not in split_data:
                print(f"  ERROR: missing key '{key}' in split '{split}'")
                ok = False

        if not ok:
            continue

        n = split_data["text"].shape[0]
        print(f"  {split}: {n} samples")
        for key, expected_dim in EXPECTED_DIMS.items():
            shape = split_data[key].shape
            print(f"    {key}: {shape}")
            if shape[-1] != expected_dim:
                print(
                    f"    ERROR: {key} dimension {shape[-1]} != {expected_dim} "
                    f"expected for CH-SIMS"
                )
                ok = False
        if n != expected_n:
            print(f"    ERROR: {n} samples != {expected_n} expected for CH-SIMS")
            ok = False

        labels = split_data["classification_labels"]
        dist = {int(c): int((labels == c).sum()) for c in sorted(set(labels.tolist()))}
        print(f"    label distribution (0=neg, 1=neu, 2=pos): {dist}")

    if not ok:
        print(
            "\n  The file does not match CH-SIMS. Note that CMU-MOSEI ships as "
            "'unaligned_50.pkl' in the same MMSA layout and is a different "
            "dataset with different splits and feature dimensions."
        )
    return ok


def sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare the CH-SIMS dataset")
    parser.add_argument("--data-root", type=str, default="data/CH-SIMS",
                        help="Root directory for CH-SIMS data")
    args = parser.parse_args()

    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)
    pkl_path = find_pickle(data_root)

    if not os.path.exists(pkl_path):
        print(f"CH-SIMS pickle not found at {pkl_path}.")
        print()
        print("Request access to the CH-SIMS pre-extracted features from the")
        print("MMSA toolkit and accept the dataset terms:")
        print("  https://github.com/thuiar/MMSA")
        print("  (see 'Datasets' -> SIMS -> unaligned_39.pkl)")
        print()
        print(f"Then place the file at: {pkl_path}")
        sys.exit(1)

    print(f"CH-SIMS pickle found at {pkl_path}")
    print("Verifying...")
    if not verify_pickle(pkl_path):
        sys.exit(1)

    print(f"\n  sha256: {sha256(pkl_path)}")
    print("\nCH-SIMS dataset is ready.")


if __name__ == "__main__":
    main()
