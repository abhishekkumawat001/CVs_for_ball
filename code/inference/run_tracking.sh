#!/bin/bash
# Run YOLO Detection + Kalman Tracking on all test videos

set -e

# Note: Activate your Python environment before running this script
# Example: conda activate your_env_name

echo "=========================================="
echo "YOLO Detection + Kalman Tracking Pipeline"
echo "=========================================="
echo ""
echo "Model: optimized3 (best performing)"
echo "Confidence: 0.1"
echo "Image size: 1280"
echo "Output: output/tracked_videos/"
echo ""
echo "Processing all 15 test videos..."
echo "=========================================="
echo ""

python inference_with_tracking.py \
    --model runs/detect/local_finetune_optimized3/weights/best.pt \
    --input-dir data/raw/25_nov_2025 \
    --output-dir output/tracked_videos \
    --conf 0.1 \
    --imgsz 1280

echo ""
echo "=========================================="
echo "✅ COMPLETE"
echo "=========================================="
echo "Outputs saved to: output/tracked_videos/"
echo "  - *_tracked.mp4: Videos with tracking visualization"
echo "  - *_trajectory.json: Trajectory data with speeds and bounces"
echo "  - processing_summary.json: Summary of all videos"
echo ""
