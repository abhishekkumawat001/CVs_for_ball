#!/bin/bash
# YOLO11 Training Script
# Ball Detection - Kaggle Dataset Pretraining

set -e

# Note: Activate your Python environment before running this script
# Example: conda activate your_env_name

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
# Configure GPUs to use (adjust based on your available GPUs)
# export CUDA_VISIBLE_DEVICES=0,1  # Uncomment to select specific GPUs

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Model
MODEL="../../models/weights/yolo11n.pt"  # Nano model - good starting point

# Dataset
DATA="../utils/kaggle_dataset.yaml"

# Training hyperparameters
EPOCHS=50          # Training epochs
BATCH=32           # Batch size (adjust based on GPU memory)
IMGSZ=640          # Input image size
WORKERS=16         # CPU workers for data loading (adjust based on available CPU cores)

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
echo "  Batch size: $BATCH"
echo "  Image size: $IMGSZ"
echo "  Optimizer: $OPTIMIZER"
echo "  GPUs: $DEVICE"
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
