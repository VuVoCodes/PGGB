#!/bin/bash
# ============================================================================
# ASGML Dataset Preparation Script
# Downloads and prepares all datasets for multimodal learning experiments
# ============================================================================
#
# Datasets:
#   1. CREMA-D    - Audio-visual emotion recognition (6 classes, ~7.4k videos)
#   2. AVE        - Audio-visual event localization (28 classes, ~4.1k videos)
#   3. Kinetics-Sounds - Human action recognition (31 classes, ~19k videos)
#
# Usage:
#   ./scripts/prepare_datasets.sh [dataset]
#
# Examples:
#   ./scripts/prepare_datasets.sh          # Prepare all datasets
#   ./scripts/prepare_datasets.sh cremad   # Only CREMA-D
#   ./scripts/prepare_datasets.sh ave      # Only AVE
#   ./scripts/prepare_datasets.sh ks       # Only Kinetics-Sounds
#
# Requirements:
#   - git, git-lfs
#   - ffmpeg (for frame extraction)
#   - gdown (pip install gdown) for Google Drive downloads
#   - yt-dlp (pip install yt-dlp) for YouTube downloads
#
# ============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    command -v git >/dev/null 2>&1 || missing+=("git")
    command -v ffmpeg >/dev/null 2>&1 || missing+=("ffmpeg")

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_error "Please install them and try again."
        exit 1
    fi

    # Check for git-lfs
    if ! git lfs version >/dev/null 2>&1; then
        log_warn "git-lfs not found. Installing..."
        if command -v brew >/dev/null 2>&1; then
            brew install git-lfs
        elif command -v apt-get >/dev/null 2>&1; then
            sudo apt-get install -y git-lfs
        else
            log_error "Please install git-lfs manually"
            exit 1
        fi
    fi

    # Check for gdown (for Google Drive)
    if ! command -v gdown >/dev/null 2>&1; then
        log_warn "gdown not found. Installing..."
        pip install gdown
    fi

    log_info "All dependencies satisfied."
}

# ============================================================================
# CREMA-D Dataset
# Source: https://github.com/CheyneyComputerScience/CREMA-D
# ============================================================================
prepare_cremad() {
    log_info "=========================================="
    log_info "Preparing CREMA-D Dataset"
    log_info "=========================================="

    local CREMAD_DIR="${DATA_DIR}/CREMA-D"

    if [ -d "$CREMAD_DIR" ] && [ -f "$CREMAD_DIR/VideoFrames/1001_DFA_ANG_XX.jpg" ]; then
        log_info "CREMA-D already prepared. Skipping."
        return
    fi

    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    # Clone repository
    if [ ! -d "CREMA-D" ]; then
        log_info "Cloning CREMA-D repository..."
        git clone https://github.com/CheyneyComputerScience/CREMA-D.git
    fi

    cd CREMA-D

    # Initialize and pull LFS files
    log_info "Pulling LFS files (this may take a while)..."
    git lfs install
    git lfs fetch --all
    git lfs checkout

    # Return to project root and run extraction script
    cd "$PROJECT_ROOT"
    log_info "Extracting video frames..."
    bash "${SCRIPT_DIR}/extract_cremad_frames.sh" "$CREMAD_DIR"

    log_info "CREMA-D preparation complete!"
    log_info "  - Audio: $CREMAD_DIR/AudioWAV/"
    log_info "  - Video frames: $CREMAD_DIR/VideoFrames/"
}

# ============================================================================
# AVE Dataset (Audio-Visual Event)
# Source: https://github.com/YapengTian/AVE-ECCV18
# Download: https://drive.google.com/open?id=1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK
# ============================================================================
prepare_ave() {
    log_info "=========================================="
    log_info "Preparing AVE Dataset"
    log_info "=========================================="

    local AVE_DIR="${DATA_DIR}/AVE"

    if [ -d "$AVE_DIR/video_frames" ] && [ -d "$AVE_DIR/audio" ]; then
        local frame_count=$(ls "$AVE_DIR/video_frames" 2>/dev/null | wc -l)
        if [ "$frame_count" -gt 100 ]; then
            log_info "AVE already prepared ($frame_count videos). Skipping."
            return
        fi
    fi

    mkdir -p "$AVE_DIR"
    cd "$AVE_DIR"

    log_info "Downloading AVE dataset from Google Drive..."
    log_info "This requires ~8GB of disk space"

    # Google Drive file ID for AVE dataset
    local GDRIVE_ID="1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK"

    if [ ! -f "AVE_Dataset.zip" ]; then
        gdown "https://drive.google.com/uc?id=${GDRIVE_ID}" -O AVE_Dataset.zip
    fi

    if [ ! -d "AVE_Dataset" ]; then
        log_info "Extracting AVE dataset..."
        unzip -q AVE_Dataset.zip
    fi

    # Clone the official repo for annotations and splits
    if [ ! -d "AVE-ECCV18" ]; then
        log_info "Cloning AVE-ECCV18 repository for annotations..."
        git clone https://github.com/YapengTian/AVE-ECCV18.git
    fi

    # Return to project root and run extraction script
    cd "$PROJECT_ROOT"
    log_info "Extracting frames and audio..."
    bash "${SCRIPT_DIR}/extract_ave_frames.sh" "$AVE_DIR"

    log_info "AVE preparation complete!"
    log_info "  - Video frames: $AVE_DIR/video_frames/"
    log_info "  - Audio: $AVE_DIR/audio/"
    log_info "  - Annotations: $AVE_DIR/annotations/"
}

# ============================================================================
# Kinetics-Sounds Dataset
# 31 classes subset of Kinetics-400
# Classes from: https://github.com/cvdfoundation/kinetics-dataset
# ============================================================================

# The 31 Kinetics-Sounds classes (from Look, Listen and Learn paper)
KINETICS_SOUNDS_CLASSES=(
    "blowing nose"
    "blowing out candles"
    "bowling"
    "chopping wood"
    "dribbling basketball"
    "laughing"
    "mowing lawn"
    "playing accordion"
    "playing bagpipes"
    "playing bass guitar"
    "playing clarinet"
    "playing drums"
    "playing guitar"
    "playing harmonica"
    "playing keyboard"
    "playing organ"
    "playing piano"
    "playing saxophone"
    "playing trombone"
    "playing trumpet"
    "playing violin"
    "playing xylophone"
    "ripping paper"
    "shoveling snow"
    "shuffling cards"
    "singing"
    "stomping grapes"
    "tap dancing"
    "tapping guitar"
    "tapping pen"
    "tickling"
)

prepare_kinetics_sounds() {
    log_info "=========================================="
    log_info "Preparing Kinetics-Sounds Dataset"
    log_info "=========================================="

    local KS_DIR="${DATA_DIR}/Kinetics-Sounds"

    # Check if already processed
    if [ -d "$KS_DIR/train" ]; then
        local video_count=$(find "$KS_DIR/train" -type d -mindepth 2 2>/dev/null | wc -l)
        if [ "$video_count" -gt 1000 ]; then
            log_info "Kinetics-Sounds already has $video_count processed videos. Skipping."
            return
        fi
    fi

    mkdir -p "$KS_DIR"
    cd "$KS_DIR"

    log_warn "=============================================="
    log_warn "Kinetics-Sounds requires downloading from YouTube."
    log_warn "This is a semi-automated process that may take hours."
    log_warn "=============================================="

    # Check for yt-dlp
    if ! command -v yt-dlp >/dev/null 2>&1; then
        log_warn "yt-dlp not found. Installing..."
        pip install yt-dlp
    fi

    # Download kinetics-400 annotations
    if [ ! -f "kinetics-400_train.csv" ]; then
        log_info "Downloading Kinetics-400 annotations..."
        wget -q "https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz" -O kinetics400.tar.gz
        tar -xzf kinetics400.tar.gz
        mv kinetics400/*.csv . 2>/dev/null || true
        rm -rf kinetics400 kinetics400.tar.gz
    fi

    # Create class list file
    log_info "Creating Kinetics-Sounds class list..."
    printf '%s\n' "${KINETICS_SOUNDS_CLASSES[@]}" > kinetics_sounds_classes.txt

    # Create download script
    log_info "Creating download script..."
    cat > download_ks_videos.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Download Kinetics-Sounds videos from YouTube."""
import os
import csv
import subprocess
from pathlib import Path

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

def download_video(youtube_id, start_time, end_time, output_path):
    """Download a video clip from YouTube."""
    if os.path.exists(output_path):
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*{start_time}-{end_time}",
        "-o", output_path,
        "--quiet",
        url
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except Exception:
        return False

def main():
    for split in ["train", "validate", "test"]:
        csv_file = f"kinetics-400_{split}.csv"
        if not os.path.exists(csv_file):
            print(f"Skipping {split}: {csv_file} not found")
            continue

        print(f"Processing {split} split...")

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                label = row.get('label', '')
                if label not in KS_CLASSES:
                    continue

                youtube_id = row.get('youtube_id', '')
                start = float(row.get('time_start', 0))
                end = float(row.get('time_end', 10))

                class_dir = Path("videos") / label.replace(" ", "_")
                class_dir.mkdir(parents=True, exist_ok=True)

                output = class_dir / f"{youtube_id}_{int(start)}_{int(end)}.mp4"

                if download_video(youtube_id, start, end, str(output)):
                    print(f"  Downloaded: {output.name}")

                if i % 100 == 0:
                    print(f"  Processed {i} entries...")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

    chmod +x download_ks_videos.py

    log_info ""
    log_info "Kinetics-Sounds setup complete!"
    log_info ""
    log_info "To download and process videos, run these commands:"
    log_info "  cd ${KS_DIR}"
    log_info "  python download_ks_videos.py                           # Step 1: Download from YouTube"
    log_info "  bash ${SCRIPT_DIR}/extract_kinetics_frames.sh ${KS_DIR}  # Step 2: Extract frames and audio"
    log_info ""
    log_warn "Note: YouTube downloads may fail for some videos (removed/private)."
    log_warn "Expect ~15-18k videos out of 19k to be available."
}

# ============================================================================
# Main
# ============================================================================
main() {
    log_info "ASGML Dataset Preparation Script"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Data directory: $DATA_DIR"
    echo ""

    check_dependencies

    local dataset="${1:-all}"

    case "$dataset" in
        cremad|crema-d|CREMA-D)
            prepare_cremad
            ;;
        ave|AVE)
            prepare_ave
            ;;
        ks|kinetics|kinetics-sounds|KS)
            prepare_kinetics_sounds
            ;;
        all)
            prepare_cremad
            prepare_ave
            prepare_kinetics_sounds
            ;;
        *)
            log_error "Unknown dataset: $dataset"
            log_info "Valid options: cremad, ave, ks, all"
            exit 1
            ;;
    esac

    echo ""
    log_info "=========================================="
    log_info "Dataset preparation complete!"
    log_info "=========================================="
    log_info ""
    log_info "Dataset locations:"
    log_info "  CREMA-D:         ${DATA_DIR}/CREMA-D"
    log_info "  AVE:             ${DATA_DIR}/AVE"
    log_info "  Kinetics-Sounds: ${DATA_DIR}/Kinetics-Sounds"
}

main "$@"
