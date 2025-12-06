#!/bin/bash
# Process all 15 cricket videos with optimized3 model

set -e

eval "$(conda shell.bash hook)"
conda activate swe

echo "=========================================="
echo "Processing All 15 Cricket Videos"
echo "Model: optimized3"
echo "=========================================="
echo ""

# Counter
count=0
total=15

# Process each video
for video in ../../data/raw/25_nov_2025/*.{mp4,mov}; do
    if [ -f "$video" ]; then
        count=$((count + 1))
        echo ""
        echo "[$count/$total] Processing: $(basename $video)"
        echo "----------------------------------------"
        
        yolo predict \
            model=../../runs/detect/local_finetune_optimized3/weights/best.pt \
            source="$video" \
            conf=0.1 \
            iou=0.5 \
            imgsz=1280 \
            max_det=5 \
            save=true \
            project=../../runs/detect \
            name=predict_optimized \
            exist_ok=True \
            line_width=5 \
            verbose=False 2>&1 | tail -5
        
        echo "✓ Completed: $(basename $video)"
    fi
done

echo ""
echo "=========================================="
echo "✅ ALL VIDEOS PROCESSED"
echo "=========================================="
echo "Output directory: runs/detect/predict_optimized/"
echo "Total videos: $count"
echo ""
