#!/usr/bin/env python3
"""Download Kinetics-Sounds videos from YouTube using yt-dlp.

Downloads only the 31 KS classes from Kinetics-400 annotations.
Processes train and val splits (test has no labels).

Usage:
    python scripts/download_kinetics_sounds.py [--workers 2] [--data-dir data/Kinetics-Sounds]
"""
import os
import csv
import subprocess
import argparse
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

KS_CLASSES = [
    "blowing nose", "blowing out candles", "bowling", "chopping wood",
    "dribbling basketball", "laughing", "mowing lawn", "playing accordion",
    "playing bagpipes", "playing bass guitar", "playing clarinet",
    "playing drums", "playing guitar", "playing harmonica", "playing keyboard",
    "playing organ", "playing piano", "playing saxophone", "playing trombone",
    "playing trumpet", "playing violin", "playing xylophone", "ripping paper",
    "shoveling snow", "shuffling cards", "singing", "stomping grapes",
    "tap dancing", "tapping guitar", "tapping pen", "tickling"
]


def download_video(youtube_id: str, start: float, end: float, output_path: str) -> bool:
    """Download a video clip from YouTube using yt-dlp."""
    # Check if already downloaded with valid size
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True

    # Remove any empty/corrupt partial file
    if os.path.exists(output_path):
        os.remove(output_path)

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp",
        "-f", "b",
        "--download-sections", f"*{int(start)}-{int(end)}",
        "-o", output_path,
        "--no-warnings",
        "--quiet",
        "--no-check-certificates",
        "--retries", "1",
        "--socket-timeout", "15",
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=90)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception:
        # Clean up any partial file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        return False


def parse_annotations(csv_path: str, split_name: str) -> list:
    """Parse K400 CSV and filter to KS classes."""
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('label', '')
            if label in KS_CLASSES:
                entries.append({
                    'label': label,
                    'youtube_id': row.get('youtube_id', ''),
                    'start': float(row.get('time_start', 0)),
                    'end': float(row.get('time_end', 10)),
                    'split': split_name,
                })
    return entries


def log(msg: str):
    """Print and flush immediately."""
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description='Download Kinetics-Sounds dataset')
    parser.add_argument('--data-dir', type=str, default='data/Kinetics-Sounds',
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of parallel downloads')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'all'],
                        help='Which split to download')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking file
    progress_file = data_dir / 'download_progress.json'
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {'downloaded': [], 'failed': []}

    downloaded_set = set(progress['downloaded'])
    failed_set = set(progress['failed'])

    # Parse annotations
    all_entries = []
    splits = ['train', 'val'] if args.split == 'all' else [args.split]

    for split in splits:
        csv_file = data_dir / f'k400_{split}.csv'
        if not csv_file.exists():
            # Try current directory
            csv_file = Path(f'k400_{split}.csv')
        if not csv_file.exists():
            log(f"Warning: k400_{split}.csv not found, skipping {split}")
            continue
        entries = parse_annotations(str(csv_file), split)
        all_entries.extend(entries)
        log(f"  {split}: {len(entries)} KS videos found in annotations")

    log(f"Total KS entries: {len(all_entries)}")
    log(f"Already downloaded: {len(downloaded_set)}")
    log(f"Previously failed: {len(failed_set)}")

    # Filter already downloaded AND previously failed
    to_download = []
    for e in all_entries:
        vid_id = f"{e['youtube_id']}_{int(e['start'])}_{int(e['end'])}"
        if vid_id not in downloaded_set and vid_id not in failed_set:
            to_download.append(e)

    log(f"Remaining to download: {len(to_download)}")

    if not to_download:
        log("Nothing left to download!")
        # Print final stats
        for split in splits:
            split_dir = data_dir / split
            if split_dir.exists():
                vids = list(split_dir.rglob('*.mp4'))
                log(f"  {split}: {len(vids)} videos on disk")
        return

    # Class distribution
    class_counts = defaultdict(int)
    for e in all_entries:
        class_counts[e['label']] += 1
    log(f"\nClass distribution ({len(class_counts)} classes):")
    for cls in sorted(class_counts, key=class_counts.get, reverse=True)[:5]:
        log(f"  {cls}: {class_counts[cls]}")
    log(f"  ... ({len(class_counts) - 5} more)")
    # Shuffle to spread classes and avoid hitting same YouTube channels sequentially
    random.seed(42)
    random.shuffle(to_download)

    log(f"\nStarting download with {args.workers} workers...\n")

    # Download — sequential to avoid YouTube rate limiting
    success = 0
    fail = 0
    total = len(to_download)

    for i, entry in enumerate(to_download):
        split_dir = 'train' if entry['split'] == 'train' else 'val'
        class_dir = data_dir / split_dir / entry['label'].replace(' ', '_')
        class_dir.mkdir(parents=True, exist_ok=True)

        vid_id = f"{entry['youtube_id']}_{int(entry['start'])}_{int(entry['end'])}"
        output = class_dir / f"{vid_id}.mp4"

        ok = download_video(entry['youtube_id'], entry['start'], entry['end'], str(output))

        if ok:
            success += 1
            progress['downloaded'].append(vid_id)
        else:
            fail += 1
            progress['failed'].append(vid_id)

        done = success + fail
        if done % 100 == 0 or done == total:
            pct = 100 * done / total
            log(f"  [{done}/{total}] ({pct:.1f}%) — {success} ok, {fail} failed")
            # Save progress periodically
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

    # Final save
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

    log(f"\nDownload complete!")
    log(f"  Success: {success}/{total} ({100*success/max(total,1):.1f}%)")
    log(f"  Failed: {fail}/{total}")
    log(f"  Total downloaded (all runs): {len(progress['downloaded'])}")

    # Count per split
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            vids = list(split_dir.rglob('*.mp4'))
            log(f"  {split}: {len(vids)} videos on disk")


if __name__ == '__main__':
    main()
