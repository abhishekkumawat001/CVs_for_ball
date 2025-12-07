#!/bin/bash
# YOLO11 Finetuning Script for Local Dataset
# Ball Detection - Domain-Specific Finetuning

set -e

# Note: Activate your Python environment before running this script
# Example: conda activate your_env_name

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
# export CUDA_VISIBLE_DEVICES=0  # Uncomment to select specific GPU

# ============================================================================
# TRAINING CONFIGURATION - FINETUNE FROM KAGGLE
# ============================================================================

# Model (pretrained from Kaggle)
MODEL="../../runs/detect/kaggle_pretrain_optimized2/weights/best.pt"

# Dataset
DATA="../utils/local_dataset.yaml"

# Training hyperparameters
EPOCHS=300         # More epochs for better convergence
BATCH=16           # Smaller batch for stability
IMGSZ=1280         # Larger size for small object detection
WORKERS=8          # Fewer workers for small dataset

# Learning rate (LOWER for finetuning)
LR0=0.0005         # Very low for finetuning (20x lower than pretraining)
LRF=0.01           # Final learning rate

# Optimizer
OPTIMIZER="AdamW"  # Keep consistent with pretraining

# Augmentation (STRONGER for small dataset to prevent overfitting)
MOSAIC=1.0         # Keep mosaic
MIXUP=0.15         # Slightly more mixup
DEGREES=15         # More rotation
TRANSLATE=0.15     # More translation
SCALE=0.7          # More scale variation
FLIPUD=0.0         # No vertical flip
FLIPLR=0.5         # Horizontal flip
HSV_H=0.02         # More color augmentation
HSV_S=0.8          
HSV_V=0.5          

# Copy-paste augmentation for small objects
COPY_PASTE=0.3     # Copy-paste ball to different locations

# Advanced settings
PATIENCE=50        # Much more patience for better convergence
SAVE_PERIOD=10     # Save less frequently
CLOSE_MOSAIC=20    # Disable mosaic in last 20 epochs

# Output
NAME="local_finetune_optimized"

# ============================================================================
# TRAINING COMMAND
# ============================================================================
echo "================================================================================"
echo "YOLO11 FINETUNING - LOCAL DATASET FROM KAGGLE WEIGHTS"
echo "================================================================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH"
echo "  Image size: $IMGSZ"
echo "  Optimizer: $OPTIMIZER"
echo "  Learning rate: $LR0 (lower for finetuning)"
echo "  GPU: device=0"
echo "  Workers: $WORKERS"
echo "  Training mode: Finetune from Kaggle pretrained weights"
echo "================================================================================"

# Check if Kaggle pretrained model exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Kaggle pretrained model not found at $MODEL"
    echo "Please run ./train_kaggle_pretrain.sh first (from scripts/training/)"
    exit 1
fi

yolo train \
    model=$MODEL \
    data=$DATA \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMGSZ \
    device=0 \
    workers=$WORKERS \
    optimizer=$OPTIMIZER \
    lr0=$LR0 \
    lrf=$LRF \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    warmup_bias_lr=0.1 \
    mosaic=$MOSAIC \
    mixup=$MIXUP \
    degrees=$DEGREES \
    translate=$TRANSLATE \
    scale=$SCALE \
    flipud=$FLIPUD \
    fliplr=$FLIPLR \
    hsv_h=$HSV_H \
    hsv_s=$HSV_S \
    hsv_v=$HSV_V \
    copy_paste=$COPY_PASTE \
    patience=$PATIENCE \
    save_period=$SAVE_PERIOD \
    close_mosaic=$CLOSE_MOSAIC \
    name=$NAME \
    exist_ok=False \
    pretrained=False \
    verbose=True \
    plots=True \
    val=True \
    cache=True

echo ""
echo "================================================================================"
echo "✅ FINETUNING COMPLETE"
echo "================================================================================"
echo "Results saved to: runs/detect/$NAME"
echo "Best weights: runs/detect/$NAME/weights/best.pt"
echo ""
echo "Next steps:"
echo "  1. Evaluate on test set:"
echo "     yolo val model=runs/detect/$NAME/weights/best.pt data=local_dataset.yaml split=test"
echo "  2. Test on video:"
echo "     yolo predict model=runs/detect/$NAME/weights/best.pt source=data/raw/25_nov_2025/1.mp4"
echo "================================================================================"
