#!/bin/bash
# Generate CSVs and trajectory videos for all test videos

set -e

# Note: Activate your Python environment before running this script
# Example: conda activate your_env_name

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
            --output-dir results \
            2>&1 | grep -E "(INFO|ERROR|✅)" | tail -5
        
        echo ""
    fi
done

echo "========================================"
echo "✅ All videos processed"
echo "========================================"
echo ""
echo "Outputs:"
echo "  Videos: results/*_tracked.mp4"
echo "  CSVs:   results/*_detections.csv"
echo "  JSON:   results/*_trajectory.json"
