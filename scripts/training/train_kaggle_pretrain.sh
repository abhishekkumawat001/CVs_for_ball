#!/bin/bash
# Optimized YOLO11 Training Script for RTX A6000 GPUs
# Ball Detection - Kaggle Dataset Pretraining

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate swe

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
# You have 3x RTX A6000 (48GB each)
# GPU 0: Available (48.6GB free) ✅
# GPU 1: In use (39GB free) - avoid
# GPU 2: Available (48.6GB free) ✅

# Select GPUs to use (GPU 0 and 2 are free)
export CUDA_VISIBLE_DEVICES=0,2

# ============================================================================
# TRAINING CONFIGURATION - OPTIMIZED FOR YOUR HARDWARE
# ============================================================================

# Model
MODEL="../../models/weights/yolo11n.pt"  # Nano model - good starting point

# Dataset
DATA="../utils/kaggle_dataset.yaml"

# Training hyperparameters
EPOCHS=50          # Increased from 30 - more training with 1778 images
BATCH=32           # Increased from 16 - you have 48GB GPU!
IMGSZ=640          # Standard for YOLO, good for mixed object sizes
WORKERS=16         # CPU workers for data loading (you have plenty of CPU)

# Learning rate
LR0=0.01           # Initial learning rate
LRF=0.01           # Final learning rate (as fraction of initial)

# Optimizer
OPTIMIZER="AdamW"  # AdamW is more stable than SGD for small datasets

# Augmentation (important for 1778 training images)
MOSAIC=1.0         # Mosaic augmentation probability
MIXUP=0.1          # MixUp augmentation probability  
DEGREES=10         # Rotation augmentation
TRANSLATE=0.1      # Translation augmentation
SCALE=0.5          # Scale augmentation
FLIPUD=0.0         # No vertical flip (balls don't appear upside down)
FLIPLR=0.5         # Horizontal flip
HSV_H=0.015        # Hue augmentation
HSV_S=0.7          # Saturation augmentation
HSV_V=0.4          # Value (brightness) augmentation

# Advanced settings
PATIENCE=15        # Early stopping patience
SAVE_PERIOD=5      # Save checkpoint every 5 epochs
CLOSE_MOSAIC=10    # Disable mosaic in last 10 epochs for better convergence

# Multi-GPU settings
DEVICE="0,1,2"       # Use GPU 0 and 2

# Output
NAME="kaggle_pretrain_optimized"

# ============================================================================
# TRAINING COMMAND
# ============================================================================
echo "================================================================================"
echo "YOLO11 TRAINING - KAGGLE DATASET PRETRAINING"
echo "================================================================================"
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: $DATA"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH (effective: $BATCH x 2 GPUs = $(($BATCH * 2)))"
echo "  Image size: $IMGSZ"
echo "  Optimizer: $OPTIMIZER"
echo "  GPUs: $DEVICE (2x RTX A6000)"
echo "  Workers: $WORKERS"
echo "================================================================================"

yolo train \
    model=$MODEL \
    data=$DATA \
    epochs=$EPOCHS \
    batch=$BATCH \
    imgsz=$IMGSZ \
    device=$DEVICE \
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
    patience=$PATIENCE \
    save_period=$SAVE_PERIOD \
    close_mosaic=$CLOSE_MOSAIC \
    name=$NAME \
    exist_ok=False \
    pretrained=True \
    verbose=True \
    plots=True \
    val=True \
    cache=True

echo ""
echo "================================================================================"
echo "✅ TRAINING COMPLETE"
echo "================================================================================"
echo "Results saved to: runs/train/$NAME"
echo "Best weights: runs/train/$NAME/weights/best.pt"
echo "Last weights: runs/train/$NAME/weights/last.pt"
echo ""
echo "Next step: Finetune on local dataset"
echo "  ./train_local_finetune.sh"
echo "================================================================================"
