#!/bin/bash
# Extract frames and audio from Kinetics-Sounds videos
# Usage: ./extract_kinetics_frames.sh /path/to/Kinetics-Sounds

KS_ROOT="${1:-data/Kinetics-Sounds}"
VIDEO_DIR="$KS_ROOT/videos"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    log_error "Video directory not found: $VIDEO_DIR"
    log_error "Please download Kinetics-Sounds videos first using download_ks_videos.py"
    exit 1
fi

# Create output directories for splits
mkdir -p "$KS_ROOT/train" "$KS_ROOT/val" "$KS_ROOT/test"

# Build split mapping from CSV files
declare -A SPLIT_MAP

log_info "Building split mapping from CSV files..."

for split_file in "$KS_ROOT/kinetics-400_train.csv" "$KS_ROOT/kinetics-400_validate.csv" "$KS_ROOT/kinetics-400_test.csv"; do
    if [ -f "$split_file" ]; then
        split_name=$(basename "$split_file" .csv)
        split_name=${split_name#kinetics-400_}

        # Map 'validate' to 'val'
        if [ "$split_name" = "validate" ]; then
            split_name="val"
        fi

        # Read CSV and extract youtube_id -> split mapping
        while IFS=, read -r label youtube_id time_start time_end split_field rest; do
            # Skip header
            if [ "$youtube_id" = "youtube_id" ]; then
                continue
            fi
            SPLIT_MAP["$youtube_id"]="$split_name"
        done < "$split_file"
    fi
done

log_info "Loaded split info for ${#SPLIT_MAP[@]} videos"

# Count total videos
TOTAL=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
COUNT=0
PROCESSED=0

log_info "Processing $TOTAL Kinetics-Sounds videos..."

# Process each class directory
for class_dir in "$VIDEO_DIR"/*/; do
    if [ ! -d "$class_dir" ]; then
        continue
    fi

    class_name=$(basename "$class_dir")

    for video in "$class_dir"*.mp4; do
        if [ ! -f "$video" ]; then
            continue
        fi

        # Get base name and extract youtube_id
        basename=$(basename "$video" .mp4)
        # Format: {youtube_id}_{start}_{end}.mp4
        youtube_id=$(echo "$basename" | cut -d'_' -f1)

        # Determine split (default to train if not found)
        split="${SPLIT_MAP[$youtube_id]:-train}"

        # Create output directory structure: split/class/video_id/
        output_dir="$KS_ROOT/$split/$class_name/$basename"
        frame_dir="$output_dir/frames"
        mkdir -p "$frame_dir"

        # Extract frames at 3 fps
        if [ ! -f "$frame_dir/frame_001.jpg" ]; then
            ffmpeg -i "$video" -vf "fps=3" -q:v 2 "$frame_dir/frame_%03d.jpg" -y 2>/dev/null
            if [ $? -eq 0 ]; then
                ((PROCESSED++))
            fi
        else
            ((PROCESSED++))
        fi

        # Extract audio as 16kHz mono WAV
        audio_output="$output_dir/audio.wav"
        if [ ! -f "$audio_output" ]; then
            ffmpeg -i "$video" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$audio_output" -y 2>/dev/null
        fi

        ((COUNT++))
        if [ $((COUNT % 500)) -eq 0 ]; then
            log_info "Processed $COUNT / $TOTAL videos ($PROCESSED successfully)"
        fi
    done
done

log_info "Done! Processed $COUNT videos ($PROCESSED successfully)"
log_info "  - Train: $(find "$KS_ROOT/train" -type d -mindepth 2 2>/dev/null | wc -l) videos"
log_info "  - Val: $(find "$KS_ROOT/val" -type d -mindepth 2 2>/dev/null | wc -l) videos"
log_info "  - Test: $(find "$KS_ROOT/test" -type d -mindepth 2 2>/dev/null | wc -l) videos"
