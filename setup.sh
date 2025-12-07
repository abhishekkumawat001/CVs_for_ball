#!/bin/bash
# Quick Setup Script for Cricket Ball Detection & Tracking

set -e

echo "Quick Setup"
echo "Cricket Ball Detection & Tracking System"
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python $python_version"

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed"
echo ""

echo ""
echo "Checking CUDA..."
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available()); print('✓ CUDA devices:', torch.cuda.device_count())" 2>/dev/null || echo "⚠ CUDA not available (CPU mode)"
echo ""


echo "Creating output directories..."
mkdir -p results
mkdir -p output/frames
mkdir -p output/trajectories
echo "Directories created"
echo ""

# Check datasets
# echo "Checking datasets..."
# if [ -d "dataset_from kaggle" ]; then
#     echo "✓ Kaggle dataset found"
# else
#     echo "⚠ Kaggle dataset not found"
# fi

# if [ -d "dataset_local" ]; then
#     echo "✓ Local dataset found"
# else
#     echo "⚠ Local dataset not found"
# fi
# echo ""

# Check base model
echo "Checking base model..."
if [ -f "models/weights/yolo11n.pt" ]; then
    echo "✓ Base model found: yolo11n.pt"
else
    echo "⚠ Base model not found"
    echo "Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt"
fi
echo ""

# Summary

echo "Setup Complete"
echo "Next steps:"
echo "  1. Run the main launcher: ./run.sh"
echo "  2. Or run pipeline directly: ./pipeline.sh"
echo ""

