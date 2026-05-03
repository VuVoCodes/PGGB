#!/usr/bin/env python3
"""
Prepare AVE dataset for training.

Extracts the AVE_Dataset.zip, processes audio into spectrograms (matching OGM-GE format),
extracts video frames, and creates train/test split files.

Expected input:  data/AVE/AVE_Dataset.zip
Expected output: data/AVE/{visual/, audio_spec/, stat.txt, my_train.txt, my_test.txt}

Usage:
    python scripts/prepare_ave.py [--data-root data/AVE] [--seed 42] [--test-ratio 0.2]
"""

import argparse
import csv
import os
import pickle
import random
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torchaudio


def parse_annotations(anno_path: str) -> list:
    """Parse AVE Annotations.txt file.

    Parameters
    ----------
    anno_path : str
        Path to Annotations.txt

    Returns
    -------
    list
        List of (category, video_id) tuples with duplicates removed.
    """
    entries = []
    seen_ids = set()

    with open(anno_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("&")
            if len(parts) < 5:
                continue
            category = parts[0]
            video_id = parts[1]
            # Skip header
            if category == "Category":
                continue
            # AVE deduplication: keep last occurrence (matching OGM-GE behavior)
            if video_id in seen_ids:
                # Remove previous entry
                entries = [(c, v) for c, v in entries if v != video_id]
            seen_ids.add(video_id)
            entries.append((category, video_id))

    return entries


def extract_audio(video_path: str, audio_path: str, sr: int = 16000) -> bool:
    """Extract audio from video using ffmpeg.

    Parameters
    ----------
    video_path : str
        Path to input MP4 file.
    audio_path : str
        Path to output WAV file.
    sr : int
        Target sample rate.

    Returns
    -------
    bool
        True if extraction succeeded.
    """
    cmd = [
        "ffmpeg", "-i", video_path,
        "-ar", str(sr), "-ac", "1",
        "-y", "-loglevel", "error",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def compute_spectrogram(audio_path: str) -> np.ndarray:
    """Compute fbank spectrogram matching OGM-GE's process_audio.py.

    Parameters
    ----------
    audio_path : str
        Path to WAV file (16kHz mono).

    Returns
    -------
    np.ndarray
        Spectrogram of shape (1024, 128).
    """
    # Use soundfile instead of torchaudio.load (avoids TorchCodec dependency)
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    waveform = torch.FloatTensor(data).unsqueeze(0)  # (1, N)
    waveform = waveform - waveform.mean()

    norm_mean = -4.503877
    norm_std = 5.141276

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    target_length = 1024
    n_frames = fbank.shape[0]
    p = target_length - n_frames

    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank.numpy()


def extract_frames(video_path: str, frame_dir: str, fps: int = 1) -> int:
    """Extract frames from video using ffmpeg.

    Parameters
    ----------
    video_path : str
        Path to input MP4 file.
    frame_dir : str
        Directory to store extracted frames.
    fps : int
        Frames per second to extract.

    Returns
    -------
    int
        Number of frames extracted.
    """
    os.makedirs(frame_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-y", "-loglevel", "error",
        os.path.join(frame_dir, "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True)
    frames = [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description="Prepare AVE dataset")
    parser.add_argument("--data-root", type=str, default="data/AVE",
                        help="Root directory containing AVE_Dataset.zip")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/test split")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fraction of data for test set")
    parser.add_argument("--fps", type=int, default=1,
                        help="Frames per second to extract from videos")
    args = parser.parse_args()

    data_root = args.data_root
    zip_path = os.path.join(data_root, "AVE_Dataset.zip")

    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found.")
        sys.exit(1)

    visual_dir = os.path.join(data_root, "visual")
    audio_spec_dir = os.path.join(data_root, "audio_spec")
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(audio_spec_dir, exist_ok=True)

    # Step 1: Extract ZIP
    print("Step 1: Extracting ZIP...")
    extract_dir = os.path.join(data_root, "AVE_Dataset")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        print(f"  Extracted to {extract_dir}")
    else:
        print(f"  Already extracted at {extract_dir}")

    # Step 2: Parse annotations
    print("Step 2: Parsing annotations...")
    anno_path = os.path.join(extract_dir, "Annotations.txt")
    entries = parse_annotations(anno_path)
    print(f"  Found {len(entries)} unique video entries")

    # Get sorted class names
    categories = sorted(set(cat for cat, _ in entries))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    print(f"  {len(categories)} classes: {categories[:5]}...")

    # Step 3: Process each video
    print("Step 3: Processing videos...")
    video_dir = os.path.join(extract_dir, "AVE")
    valid_entries = []
    failed = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (category, video_id) in enumerate(entries):
            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            if not os.path.exists(video_path):
                failed.append((video_id, "video not found"))
                continue

            # Check if already processed
            spec_path = os.path.join(audio_spec_dir, f"{video_id}.pkl")
            frame_dir = os.path.join(visual_dir, video_id)
            frames_exist = os.path.exists(frame_dir) and len(
                [f for f in os.listdir(frame_dir) if f.endswith(".jpg")]
            ) > 0
            spec_exists = os.path.exists(spec_path)

            if frames_exist and spec_exists:
                valid_entries.append((category, video_id))
                if (i + 1) % 500 == 0:
                    print(f"  [{i+1}/{len(entries)}] {video_id} (cached)")
                continue

            # Extract audio
            wav_path = os.path.join(tmp_dir, f"{video_id}.wav")
            if not spec_exists:
                if not extract_audio(video_path, wav_path):
                    failed.append((video_id, "audio extraction failed"))
                    continue

                # Compute spectrogram
                try:
                    spec = compute_spectrogram(wav_path)
                    with open(spec_path, "wb") as f:
                        pickle.dump(spec, f)
                except Exception as e:
                    failed.append((video_id, f"spectrogram failed: {e}"))
                    continue

                # Clean up wav
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            # Extract frames
            if not frames_exist:
                n_frames = extract_frames(video_path, frame_dir, fps=args.fps)
                if n_frames == 0:
                    failed.append((video_id, "no frames extracted"))
                    continue

            valid_entries.append((category, video_id))

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(entries)}] {video_id} OK")

    print(f"  Processed: {len(valid_entries)} valid, {len(failed)} failed")
    if failed:
        print(f"  Failed examples: {failed[:5]}")

    # Step 4: Create stat.txt
    print("Step 4: Creating stat.txt...")
    stat_path = os.path.join(data_root, "stat.txt")
    with open(stat_path, "w") as f:
        writer = csv.writer(f)
        for cat in categories:
            writer.writerow([cat])
    print(f"  Written {len(categories)} classes to {stat_path}")

    # Step 5: Create train/test splits
    print("Step 5: Creating train/test splits...")
    random.seed(args.seed)

    # Group by category for stratified split
    cat_videos = defaultdict(list)
    for cat, vid in valid_entries:
        cat_videos[cat].append(vid)

    train_entries = []
    test_entries = []

    for cat in categories:
        videos = cat_videos.get(cat, [])
        random.shuffle(videos)
        n_test = max(1, int(len(videos) * args.test_ratio))
        test_vids = videos[:n_test]
        train_vids = videos[n_test:]

        cat_idx = cat_to_idx[cat]
        for vid in train_vids:
            train_entries.append((cat_idx, vid))
        for vid in test_vids:
            test_entries.append((cat_idx, vid))

    # Shuffle within splits
    random.shuffle(train_entries)
    random.shuffle(test_entries)

    train_path = os.path.join(data_root, "my_train.txt")
    test_path = os.path.join(data_root, "my_test.txt")

    with open(train_path, "w") as f:
        writer = csv.writer(f)
        for cat_idx, vid in train_entries:
            writer.writerow([cat_idx, vid])

    with open(test_path, "w") as f:
        writer = csv.writer(f)
        for cat_idx, vid in test_entries:
            writer.writerow([cat_idx, vid])

    print(f"  Train: {len(train_entries)} samples → {train_path}")
    print(f"  Test:  {len(test_entries)} samples → {test_path}")

    # Summary
    print("\n=== AVE Dataset Preparation Complete ===")
    print(f"  Classes: {len(categories)}")
    print(f"  Total valid: {len(valid_entries)}")
    print(f"  Train/Test: {len(train_entries)}/{len(test_entries)}")
    print(f"  Visual: {visual_dir}")
    print(f"  Audio specs: {audio_spec_dir}")
    print(f"  Splits: {train_path}, {test_path}")
    print(f"  Classes: {stat_path}")

    # Per-class counts
    print("\n  Per-class distribution:")
    for cat in categories:
        n_train = sum(1 for c, _ in train_entries if c == cat_to_idx[cat])
        n_test = sum(1 for c, _ in test_entries if c == cat_to_idx[cat])
        print(f"    {cat}: {n_train} train / {n_test} test")


if __name__ == "__main__":
    main()
