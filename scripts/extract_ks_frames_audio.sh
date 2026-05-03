#!/bin/bash
# Extract frames and audio from downloaded Kinetics-Sounds videos
#
# Input structure:  data/Kinetics-Sounds/{train,val}/class_name/video_id.mp4
# Output structure: data/Kinetics-Sounds/{train,val}/class_name/video_id/frames/frame_001.jpg + audio.wav
#
# Usage: bash scripts/extract_ks_frames_audio.sh [data/Kinetics-Sounds]

set -uo pipefail

KS_ROOT="${1:-data/Kinetics-Sounds}"
FPS=3

echo "Extracting frames (${FPS} FPS) and audio (16kHz mono) from Kinetics-Sounds..."
echo "Root: $KS_ROOT"

TOTAL=$(find "$KS_ROOT/train" "$KS_ROOT/val" -name "*.mp4" 2>/dev/null | wc -l)
echo "Total videos: $TOTAL"

COUNT=0
OK=0
FAIL=0

for split in train val; do
    split_dir="$KS_ROOT/$split"
    [ -d "$split_dir" ] || continue

    for mp4 in $(find "$split_dir" -name "*.mp4" -type f); do
        video_id=$(basename "$mp4" .mp4)
        class_dir=$(dirname "$mp4")
        out_dir="$class_dir/$video_id"
        frame_dir="$out_dir/frames"

        # Skip if already extracted
        if [ -d "$frame_dir" ] && [ -f "$out_dir/audio.wav" ] && [ "$(find "$frame_dir" -name '*.jpg' 2>/dev/null | wc -l)" -gt 0 ]; then
            ((COUNT++))
            ((OK++))
            continue
        fi

        mkdir -p "$frame_dir"

        # Extract frames
        ffmpeg -i "$mp4" -vf "fps=$FPS" -q:v 2 "$frame_dir/frame_%03d.jpg" -y -loglevel error 2>/dev/null

        # Extract audio
        ffmpeg -i "$mp4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$out_dir/audio.wav" -y -loglevel error 2>/dev/null

        # Check success
        if [ -f "$out_dir/audio.wav" ] && [ "$(find "$frame_dir" -name '*.jpg' 2>/dev/null | wc -l)" -gt 0 ]; then
            ((OK++))
        else
            ((FAIL++))
        fi

        ((COUNT++))
        if [ $((COUNT % 500)) -eq 0 ]; then
            echo "  [$COUNT/$TOTAL] $OK ok, $FAIL fail"
        fi
    done
done

echo ""
echo "Extraction complete!"
echo "  Processed: $COUNT"
echo "  Success: $OK"
echo "  Failed: $FAIL"
echo "  Train videos: $(find "$KS_ROOT/train" -name "audio.wav" 2>/dev/null | wc -l)"
echo "  Val videos: $(find "$KS_ROOT/val" -name "audio.wav" 2>/dev/null | wc -l)"
