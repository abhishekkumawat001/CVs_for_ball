#!/bin/bash
# Generate CSVs and trajectory videos for all test videos

set -e

eval "$(conda shell.bash hook)"
conda activate swe

echo "========================================"
echo "Generating CSVs and Trajectory Videos"
echo "========================================"
echo ""

count=0
total=15

for video in data/raw/25_nov_2025/*.{mp4,mov}; do
    if [ -f "$video" ]; then
        count=$((count + 1))
        video_name=$(basename "$video")
        
        echo "[$count/$total] Processing: $video_name"
        
        python inference_with_tracking.py \
            --video "$video" \
            --output-dir output/tracked_videos \
            2>&1 | grep -E "(INFO|ERROR|✅)" | tail -5
        
        echo ""
    fi
done

echo "========================================"
echo "✅ All videos processed"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Videos: output/tracked_videos/*_tracked.mp4"
echo "  CSVs:   output/tracked_videos/*_detections.csv"
echo "  JSON:   output/tracked_videos/*_trajectory.json"
