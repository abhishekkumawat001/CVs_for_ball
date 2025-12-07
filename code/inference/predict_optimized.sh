#!/bin/bash
# Optimized YOLO11 Prediction Script for Ball Detection
# With improved parameters for small object detection

set -e

# Note: Activate your Python environment before running this script
# Example: conda activate your_env_name

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model
MODEL="${1:-../../runs/detect/local_finetune_optimized3/weights/best.pt}"

# Video source
SOURCE="${2:-../../data/raw/25_nov_2025/1.mp4}"

# Detection parameters (optimized for small cricket balls)
CONF=0.1           # Low confidence threshold to catch more balls
IOU=0.5            # IoU threshold for NMS
IMGSZ=1280         # Large image size for small objects (2x improvement)
MAX_DET=1          # Max detections per image (reduced for cleaner output)

# Output settings
SAVE=True
PROJECT="../../runs/detect"
NAME="predict_optimized"

# ============================================================================
# PREDICTION COMMAND
# ============================================================================
echo "================================================================================"
echo "YOLO11 BALL DETECTION - OPTIMIZED PREDICTION"
echo "================================================================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Source: $SOURCE"
echo "  Confidence threshold: $CONF (low for better recall)"
echo "  Image size: $IMGSZ (2x for small objects)"
echo "  IoU threshold: $IOU"
echo "================================================================================"

yolo predict \
    model=$MODEL \
    source=$SOURCE \
    conf=$CONF \
    iou=$IOU \
    imgsz=$IMGSZ \
    max_det=$MAX_DET \
    save=$SAVE \
    project=$PROJECT \
    name=$NAME \
    exist_ok=True \
    verbose=True \
    show_labels=True \
    show_conf=True \
    line_width=2

echo ""
echo "================================================================================"
echo "✅ PREDICTION COMPLETE"
echo "================================================================================"
echo "Results saved to: $PROJECT/$NAME"
# YOLO saves videos as .avi format
BASENAME=$(basename "$SOURCE")
VIDEONAME="${BASENAME%.*}.avi"
echo "Video output: $PROJECT/$NAME/$VIDEONAME"
echo "================================================================================"
