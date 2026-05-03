#!/bin/bash
# Extract middle frames from CREMA-D FLV videos
# Usage: ./extract_cremad_frames.sh /path/to/CREMA-D

CREMAD_ROOT="${1:-data/CREMA-D}"
VIDEO_DIR="$CREMAD_ROOT/VideoFlash"
OUTPUT_DIR="$CREMAD_ROOT/VideoFrames"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total files
TOTAL=$(ls "$VIDEO_DIR"/*.flv 2>/dev/null | wc -l)
COUNT=0

echo "Extracting frames from $TOTAL videos..."

for flv in "$VIDEO_DIR"/*.flv; do
    if [ -f "$flv" ]; then
        # Get base name without extension
        basename=$(basename "$flv" .flv)
        output="$OUTPUT_DIR/${basename}.jpg"

        # Skip if already extracted
        if [ -f "$output" ]; then
            ((COUNT++))
            continue
        fi

        # Extract middle frame (at 50% of video)
        # -ss 0.5 seeks to 0.5 seconds which is roughly middle for short clips
        ffmpeg -i "$flv" -vf "select=eq(n\,5)" -vframes 1 -q:v 2 "$output" -y 2>/dev/null

        ((COUNT++))
        if [ $((COUNT % 500)) -eq 0 ]; then
            echo "Processed $COUNT / $TOTAL"
        fi
    fi
done

echo "Done! Extracted $COUNT frames to $OUTPUT_DIR"
