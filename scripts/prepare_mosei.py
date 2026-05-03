#!/usr/bin/env python3
"""
Prepare CMU-MOSEI dataset for training.

Downloads the pre-processed MOSEI sentiment data pickle file from the
CMU Multimodal SDK and places it in the data directory.

The pickle contains pre-extracted features:
  - text: GloVe embeddings (300d)
  - audio: COVAREP features (74d)
  - vision: FACET features (35d)
  - labels: continuous sentiment (-3 to +3)

Reference:
Zadeh et al., "Multimodal Language Analysis in the Wild:
CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph" (ACL 2018)

Usage:
    python scripts/prepare_mosei.py [--data-root data/MOSEI]
"""

import argparse
import os
import pickle
import sys


def verify_pickle(pkl_path: str) -> bool:
    """Verify the MOSEI pickle file has expected structure.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle file.

    Returns
    -------
    bool
        True if valid.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        required_splits = ["train", "valid", "test"]
        required_keys = ["vision", "text", "audio", "labels"]

        for split in required_splits:
            if split not in data:
                print(f"  ERROR: Missing split '{split}'")
                return False
            for key in required_keys:
                if key not in data[split]:
                    print(f"  ERROR: Missing key '{key}' in split '{split}'")
                    return False

        # Print shapes
        for split in required_splits:
            print(f"  {split}:")
            for key in required_keys:
                shape = data[split][key].shape
                print(f"    {key}: {shape}")

        # Print label statistics
        for split in required_splits:
            labels = data[split]["labels"].flatten()
            neg = (labels <= -0.5).sum()
            neu = ((labels > -0.5) & (labels < 0.5)).sum()
            pos = (labels >= 0.5).sum()
            print(f"  {split} label dist: neg={neg}, neu={neu}, pos={pos}")

        return True

    except Exception as e:
        print(f"  ERROR loading pickle: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Prepare CMU-MOSEI dataset")
    parser.add_argument("--data-root", type=str, default="data/MOSEI",
                        help="Root directory for MOSEI data")
    args = parser.parse_args()

    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)

    pkl_path = os.path.join(data_root, "mosei_senti_data.pkl")

    # Check if already exists
    if os.path.exists(pkl_path):
        print(f"MOSEI pickle already exists at {pkl_path}")
        print("Verifying...")
        if verify_pickle(pkl_path):
            print("\nMOSEI dataset is ready!")
            return
        else:
            print("\nPickle exists but is invalid. Please re-download.")
            sys.exit(1)

    # Try to download using CMU Multimodal SDK
    print("MOSEI pickle not found. Attempting download...")
    print()
    print("The CMU-MOSEI pre-processed pickle must be downloaded manually.")
    print("Please follow one of these methods:")
    print()
    print("Method 1: CMU Multimodal SDK")
    print("  pip install mmsdk")
    print("  python -c \"")
    print("  from mmsdk import mmdatasdk")
    print("  mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, 'data/MOSEI/raw')")
    print("  \"")
    print()
    print("Method 2: Direct download (aligned features)")
    print("  Download mosei_senti_data.pkl from:")
    print("  https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK")
    print("  or Google Drive links in the SDK repository README")
    print()
    print(f"Place the file at: {pkl_path}")
    print()

    # Try automated download via SDK
    try:
        print("Attempting automated download via mmsdk...")
        from mmsdk import mmdatasdk  # noqa: F401
        print("  mmsdk is installed. However, the aligned pickle requires")
        print("  manual processing. Please use the SDK scripts to create")
        print("  the aligned pickle file.")
    except ImportError:
        print("  mmsdk not installed. Install with: pip install mmsdk")

    # Try alternate: download from known mirror
    print()
    print("Attempting download from known sources...")

    # The MOSEI aligned data is commonly shared via Google Drive
    # We'll try gdown if available
    try:
        import gdown
        # Common Google Drive ID for mosei_senti_data.pkl
        # This is the aligned version used by most papers
        file_id = "1I85S_d2RNMsMjNRMJSl1AiT4Pqe-FiAi"
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"  Downloading from Google Drive (file_id={file_id})...")
        gdown.download(url, pkl_path, quiet=False)

        if os.path.exists(pkl_path):
            print("  Download complete. Verifying...")
            if verify_pickle(pkl_path):
                print("\nMOSEI dataset is ready!")
                return
            else:
                print("  Downloaded file is invalid.")
                os.remove(pkl_path)
    except ImportError:
        print("  gdown not installed. Install with: pip install gdown")
    except Exception as e:
        print(f"  Download failed: {e}")

    print()
    print(f"Please download mosei_senti_data.pkl manually and place it at:")
    print(f"  {os.path.abspath(pkl_path)}")
    sys.exit(1)


if __name__ == "__main__":
    main()
