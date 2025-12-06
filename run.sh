#!/bin/bash
# Unified Launcher for Cricket Ball Detection & Tracking
# Simple interface to run all operations

set -e

clear
echo -e "Cricket Ball Detection & Tracking System"

# Function to show menu
show_menu() {
    echo "Select Operation:"
    echo ""
    echo "  TRAINING:"
    echo "    1) Train complete pipeline (Kaggle → Local) or (Optional)"
    echo "    2) Train Kaggle pretrain only"
    echo "    3) Train local finetune only"
    echo "    Optional: Any pretrain model can be used like yolov5, yolov8, yolov11 from ultralytics"
    echo ""
    echo "  INFERENCE:"
    echo "    4) Process all videos (tracked + CSVs) [~4 min] "
    echo "    5) Process single video"
    echo ""
    echo "  UTILITIES:"
    echo "    6) Check repository status"
    echo "    7) View outputs"
    echo ""
    echo "    0) Exit"
    echo ""
    echo -ne "Enter choice [0-7]:"
}

# Training functions
train_complete() {
    echo -e "Starting complete training pipeline..."
    cd code/training
    echo -e "[1/2] Kaggle pretrain"
    ./train_kaggle_pretrain.sh
    echo -e "[2/2] Local finetune"
    ./train_local_finetune.sh
    cd ../..
    echo -e "Training complete!"
}

train_kaggle() {
    echo -e "Training Kaggle pretrain model..."
    cd code/training
    ./train_kaggle_pretrain.sh
    cd ../..
    echo -e "$✅ Kaggle training complete!"
}

train_local() {
    echo -e "Training local finetune model..."
    cd code/training
    ./train_local_finetune.sh
    cd ../..
    echo -e "✅ Local training complete!"
}

# Inference functions
process_all() {
    echo -e "Processing all videos with tracking.."
    cd code/inference
    ./generate_all_csvs.sh
    cd ../..
    echo -e "✅ All videos processed!"
    echo ""
    echo -e "Outputs saved to: output/tracked_videos/"
    echo "  - 15 tracked videos (*_tracked.mp4)"
    echo "  - 15 CSV files (*_detections.csv)"
    echo "  - 15 trajectory JSONs (*_trajectory.json)"
}

process_single() {
    echo ""
    echo -e "Available videos:"
    ls data/raw/25_nov_2025/ | nl
    echo ""
    echo -ne "Enter video filename: "
    read video_name
    
    if [ -f "data/raw/25_nov_2025/$video_name" ]; then
        echo -e "Processing $video_name..."
        cd code/inference
        python inference_with_tracking.py --video "../../data/raw/25_nov_2025/$video_name"
        cd ../..
        echo -e "✅ Video processed!"
    else
        echo -e "Error: Video not found"
    fi
}

# Utility functions
check_status() {
    echo -e "Repository Status:"
    echo ""
    
    # Models
    echo -e "Models:"
    if [ -f "models/weights/yolo11n.pt" ]; then
        echo "  ✅ Base model: yolo11n.pt (5.4 MB)"
    else
        echo "  ❌ Base model not found"
    fi
    
    if [ -f "runs/detect/kaggle_pretrain_optimized2/weights/best.pt" ]; then
        echo "  ✅ Kaggle pretrained model"
    else
        echo "  ⚠️  Kaggle pretrained model not found (run training step 1)"
    fi
    
    if [ -f "runs/detect/local_finetune_optimized3/weights/best.pt" ]; then
        echo "  ✅ Production model: local_finetune_optimized3"
    else
        echo "  ⚠️  Production model not found (run training step 2)"
    fi
    
    # Data
    echo ""
    echo -e "Datasets:"
    if [ -d "dataset_from kaggle" ]; then
        echo "  ✅ Kaggle dataset"
    fi
    if [ -d "dataset_local" ]; then
        echo "  ✅ Local dataset"
    fi
    
    # Test videos
    echo ""
    echo -e "Test Videos:"
    video_count=$(ls data/raw/25_nov_2025/*.{mp4,mov} 2>/dev/null | wc -l)
    echo "  ✅ $video_count videos in data/raw/25_nov_2025/"
    
    # Outputs
    echo ""
    echo -e "Outputs:"
    if [ -d "output/tracked_videos" ]; then
        tracked_count=$(ls output/tracked_videos/*_tracked.mp4 2>/dev/null | wc -l)
        csv_count=$(ls output/tracked_videos/*_detections.csv 2>/dev/null | wc -l)
        json_count=$(ls output/tracked_videos/*_trajectory.json 2>/dev/null | wc -l)
        echo "  📹 Tracked videos: $tracked_count"
        echo "  📊 CSV files: $csv_count"
        echo "  📈 Trajectory JSONs: $json_count"
    else
        echo "  ⚠️  No outputs yet (run inference)"
    fi
    
    echo ""
}

view_outputs() {
    echo -e "Output Directory: $(pwd)/output/tracked_videos/"
    echo ""
    if [ -d "output/tracked_videos" ]; then
        echo -e "Tracked Videos:"
        ls -lh output/tracked_videos/*_tracked.mp4 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
        echo -e "CSV Files:"
        ls -lh output/tracked_videos/*_detections.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
        echo -e "Location: $(pwd)/output/tracked_videos/"
    else
        echo "  No outputs yet. Run inference first (option 4)."
    fi
    echo ""
}

# Main loop
while true; do
    show_menu
    read choice
    echo ""
    
    case $choice in
        1)
            train_complete
            ;;
        2)
            train_kaggle
            ;;
        3)
            train_local
            ;;
        4)
            process_all
            ;;
        5)
            process_single
            ;;
        6)
            check_status
            ;;
        7)
            view_outputs
            ;;
        0)
            echo -e "Goodbye!"
            exit 0
            ;;
        *)
            echo -e "Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    echo -e "Press Enter to continue..."
    read
    clear
    echo -e "╔════════════════════════════════════════════════════════════╗"
    echo -e "║     Cricket Ball Detection & Tracking System              ║"
    echo -e "╚════════════════════════════════════════════════════════════╝"
    echo ""
done
