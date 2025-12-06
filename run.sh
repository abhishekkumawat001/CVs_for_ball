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
    cd scripts/training
    echo -e "[1/2] Kaggle pretrain"
    ./train_kaggle_pretrain.sh
    echo -e "[2/2] Local finetune"
    ./train_local_finetune.sh
    cd ../..
    echo -e "Training complete!"
}

train_kaggle() {
    echo -e "${GREEN}Training Kaggle pretrain model...${NC}"
    cd scripts/training
    ./train_kaggle_pretrain.sh
    cd ../..
    echo -e "${GREEN}✅ Kaggle training complete!${NC}"
}

train_local() {
    echo -e "${GREEN}Training local finetune model...${NC}"
    cd scripts/training
    ./train_local_finetune.sh
    cd ../..
    echo -e "${GREEN}✅ Local training complete!${NC}"
}

# Inference functions
process_all() {
    echo -e "${GREEN}Processing all videos with tracking...${NC}"
    cd scripts/inference
    ./generate_all_csvs.sh
    cd ../..
    echo -e "${GREEN}✅ All videos processed!${NC}"
    echo ""
    echo -e "${BLUE}Outputs saved to: output/tracked_videos/${NC}"
    echo "  - 15 tracked videos (*_tracked.mp4)"
    echo "  - 15 CSV files (*_detections.csv)"
    echo "  - 15 trajectory JSONs (*_trajectory.json)"
}

process_single() {
    echo ""
    echo -e "${YELLOW}Available videos:${NC}"
    ls data/raw/25_nov_2025/ | nl
    echo ""
    echo -ne "${YELLOW}Enter video filename: ${NC}"
    read video_name
    
    if [ -f "data/raw/25_nov_2025/$video_name" ]; then
        echo -e "${GREEN}Processing $video_name...${NC}"
        cd scripts/inference
        python inference_with_tracking.py --video "../../data/raw/25_nov_2025/$video_name"
        cd ../..
        echo -e "${GREEN}✅ Video processed!${NC}"
    else
        echo -e "${RED}Error: Video not found${NC}"
    fi
}

# Utility functions
check_status() {
    echo -e "${BLUE}Repository Status:${NC}"
    echo ""
    
    # Models
    echo -e "${YELLOW}Models:${NC}"
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
    echo -e "${YELLOW}Datasets:${NC}"
    if [ -d "dataset_from kaggle" ]; then
        echo "  ✅ Kaggle dataset"
    fi
    if [ -d "dataset_local" ]; then
        echo "  ✅ Local dataset"
    fi
    
    # Test videos
    echo ""
    echo -e "${YELLOW}Test Videos:${NC}"
    video_count=$(ls data/raw/25_nov_2025/*.{mp4,mov} 2>/dev/null | wc -l)
    echo "  ✅ $video_count videos in data/raw/25_nov_2025/"
    
    # Outputs
    echo ""
    echo -e "${YELLOW}Outputs:${NC}"
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
    echo -e "${BLUE}Output Directory:${NC}"
    echo ""
    if [ -d "output/tracked_videos" ]; then
        echo -e "${YELLOW}Tracked Videos:${NC}"
        ls -lh output/tracked_videos/*_tracked.mp4 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
        echo -e "${YELLOW}CSV Files:${NC}"
        ls -lh output/tracked_videos/*_detections.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
        echo ""
        echo -e "${YELLOW}Location:${NC} $(pwd)/output/tracked_videos/"
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
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     Cricket Ball Detection & Tracking System              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
done
