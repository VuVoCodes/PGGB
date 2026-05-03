#!/bin/bash
# Extract frames and audio from AVE videos
# Usage: ./extract_ave_frames.sh /path/to/AVE

AVE_ROOT="${1:-data/AVE}"
VIDEO_DIR="$AVE_ROOT/AVE_Dataset"
FRAME_DIR="$AVE_ROOT/video_frames"
AUDIO_DIR="$AVE_ROOT/audio"
ANNO_DIR="$AVE_ROOT/annotations"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if video directory exists
if [ ! -d "$VIDEO_DIR" ]; then
    log_warn "Video directory not found: $VIDEO_DIR"
    log_warn "Please download AVE dataset first."
    exit 1
fi

# Create output directories
mkdir -p "$FRAME_DIR" "$AUDIO_DIR" "$ANNO_DIR"

# Copy annotations from AVE-ECCV18 repo if available
if [ -d "$AVE_ROOT/AVE-ECCV18/data" ]; then
    log_info "Copying annotations..."
    cp "$AVE_ROOT/AVE-ECCV18/data/"*.txt "$ANNO_DIR/" 2>/dev/null || true
fi

# Count total videos
TOTAL=$(ls "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | tr -d ' ')
COUNT=0

log_info "Extracting frames and audio from $TOTAL AVE videos..."

for video in "$VIDEO_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        # Get base name without extension
        basename=$(basename "$video" .mp4)

        # Create frame directory for this video
        video_frame_dir="$FRAME_DIR/$basename"
        mkdir -p "$video_frame_dir"

        # Extract frames at 1 fps (10 frames for 10-second videos)
        if [ ! -f "$video_frame_dir/frame_001.jpg" ]; then
            ffmpeg -i "$video" -vf "fps=1" -q:v 2 "$video_frame_dir/frame_%03d.jpg" -y 2>/dev/null
        fi

        # Extract audio as 16kHz mono WAV
        audio_output="$AUDIO_DIR/${basename}.wav"
        if [ ! -f "$audio_output" ]; then
            ffmpeg -i "$video" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$audio_output" -y 2>/dev/null
        fi

        ((COUNT++))
        if [ $((COUNT % 100)) -eq 0 ]; then
            log_info "Processed $COUNT / $TOTAL videos"
        fi
    fi
done

log_info "Done! Extracted from $COUNT videos"
log_info "  - Frames: $FRAME_DIR ($(ls "$FRAME_DIR" 2>/dev/null | wc -l) videos)"
log_info "  - Audio: $AUDIO_DIR ($(ls "$AUDIO_DIR"/*.wav 2>/dev/null | wc -l) files)"
