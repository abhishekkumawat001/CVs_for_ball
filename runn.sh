#!/bin/bash
# Simplified Launcher for Cricket Ball Detection & Tracking
# Demo/Inference Mode - For showcasing to employers

set -e

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     Cricket Ball Detection & Tracking System              ║"
echo "║              Demo/Inference Mode                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Google Drive IDs
MODEL_GDRIVE_ID="16SIU089-Eg6xTU8YcadvqNcBxxEs_Z_P"
VIDEOS_GDRIVE_ID="1A_R0eAUW1ZP4l866O3iBnv2rblYNvTQH"
MODEL_PATH="runs/detect/local_finetune_optimized3/weights/best.pt"

# Function to show menu
show_menu() {
    echo "Select Operation:"
    echo ""
    echo "  SETUP (First Time):"
    echo "    1) Download trained model (best.pt) from Google Drive"
    echo "    2) Download sample results videos (optional)"
    echo ""
    echo "  INFERENCE (Main Demo):"
    echo "    3) Process all videos (generate tracked videos + CSVs) ⭐"
    echo "    4) Process single video"
    echo "    5) View sample annotations (CSV format)"
    echo ""
    echo "  UTILITIES:"
    echo "    6) Check system status"
    echo "    7) View output results"
    echo ""
    echo "    0) Exit"
    echo ""
    echo -ne "Enter choice [0-7]: "
}

# Download model
download_model() {
    echo -e "${YELLOW}Downloading trained model (best.pt)...${NC}"
    echo ""
    
    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown..."
        pip install -q gdown
    fi
    
    # Create directory
    mkdir -p runs/detect/local_finetune_optimized3/weights
    
    # Download model
    echo "Downloading from Google Drive (5.4 MB)..."
    gdown ${MODEL_GDRIVE_ID} -O ${MODEL_PATH}
    
    if [ -f "${MODEL_PATH}" ]; then
        echo -e "${GREEN}✅ Model downloaded successfully!${NC}"
        echo "Location: ${MODEL_PATH}"
    else
        echo -e "${RED}❌ Download failed. Please download manually from:${NC}"
        echo "https://drive.google.com/file/d/${MODEL_GDRIVE_ID}/view"
    fi
}

# Download sample results
download_results() {
    echo -e "${YELLOW}Downloading sample results videos (123 MB)...${NC}"
    echo ""
    
    # Check if gdown is installed
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown..."
        pip install -q gdown
    fi
    
    mkdir -p results
    
    echo "Downloading from Google Drive..."
    gdown ${VIDEOS_GDRIVE_ID} -O results/edgefleet_results_videos.tar.gz
    
    if [ -f "results/edgefleet_results_videos.tar.gz" ]; then
        echo "Extracting archive..."
        tar -xzf results/edgefleet_results_videos.tar.gz -C results
        rm results/edgefleet_results_videos.tar.gz
        
        # Fix directory structure if needed
        if [ -d "results/results" ]; then
            mv results/results/* results/
            rm -r results/results
        fi
        
        echo -e "${GREEN}✅ Sample results downloaded and extracted!${NC}"
        echo "Location: results/"
    else
        echo -e "${RED}❌ Download failed. Please download manually from:${NC}"
        echo "https://drive.google.com/file/d/${VIDEOS_GDRIVE_ID}/view"
    fi
}

# Process all videos
process_all() {
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo -e "${RED}❌ Model not found!${NC}"
        echo "Please download the model first (Option 1)"
        return 1
    fi
    
    # Check if videos exist
    video_count=$(find data/raw/25_nov_2025 -type f \( -name "*.mp4" -o -name "*.mov" \) 2>/dev/null | wc -l)
    if [ "$video_count" -eq 0 ]; then
        echo -e "${RED}❌ No test videos found in data/raw/25_nov_2025/${NC}"
        echo "Please add your test videos to process."
        return 1
    fi
    
    echo -e "${YELLOW}Processing all videos with tracking...${NC}"
    echo "Found $video_count videos"
    echo ""
    
    cd code/inference
    ./generate_all_csvs.sh
    cd ../..
    
    echo ""
    echo -e "${GREEN}✅ All videos processed successfully!${NC}"
    echo ""
    echo "Outputs saved to: results/"
    echo "  - Tracked videos (*_tracked.mp4)"
    echo "  - Detection CSVs (*_detections.csv)"
    echo "  - Trajectory JSONs (*_trajectory.json)"
}

# Process single video
process_single() {
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo -e "${RED}❌ Model not found!${NC}"
        echo "Please download the model first (Option 1)"
        return 1
    fi
    
    echo ""
    echo -e "${YELLOW}Available videos:${NC}"
    
    if [ ! -d "data/raw/25_nov_2025" ] || [ -z "$(ls -A data/raw/25_nov_2025)" ]; then
        echo -e "${RED}No videos found in data/raw/25_nov_2025/${NC}"
        return 1
    fi
    
    ls data/raw/25_nov_2025/ | nl
    echo ""
    echo -ne "Enter video filename: "
    read video_name
    
    if [ -f "data/raw/25_nov_2025/$video_name" ]; then
        echo -e "${YELLOW}Processing $video_name...${NC}"
        cd code/inference
        python inference_with_tracking.py --video "../../data/raw/25_nov_2025/$video_name"
        cd ../..
        echo -e "${GREEN}✅ Video processed successfully!${NC}"
    else
        echo -e "${RED}❌ Error: Video not found${NC}"
    fi
}

# View sample annotations
view_annotations() {
    echo -e "${YELLOW}Sample CSV Annotation Format:${NC}"
    echo ""
    echo "Format: frame,x,y,visible"
    echo "  - frame: Frame number (1-indexed)"
    echo "  - x,y: Ball centroid coordinates"
    echo "  - visible: 1 (detected), 0 (not detected, x=-1, y=-1)"
    echo ""
    
    if [ -f "annotations/1_detections.csv" ]; then
        echo -e "${YELLOW}Example from annotations/1_detections.csv:${NC}"
        head -10 annotations/1_detections.csv
        echo "..."
        echo ""
        total_frames=$(tail -n +2 annotations/1_detections.csv | wc -l)
        echo "Total frames in this video: $total_frames"
    else
        echo -e "${RED}No sample annotations found.${NC}"
    fi
}

# Check status
check_status() {
    echo -e "${YELLOW}System Status:${NC}"
    echo ""
    
    # Model status
    echo "📦 Model:"
    if [ -f "${MODEL_PATH}" ]; then
        model_size=$(du -h "${MODEL_PATH}" | cut -f1)
        echo -e "  ${GREEN}✅ Production model (best.pt): $model_size${NC}"
        echo "     Location: ${MODEL_PATH}"
    else
        echo -e "  ${RED}❌ Model not found${NC}"
        echo "     Run Option 1 to download from Google Drive"
    fi
    
    echo ""
    
    # Test videos
    echo "🎥 Test Videos:"
    if [ -d "data/raw/25_nov_2025" ]; then
        video_count=$(find data/raw/25_nov_2025 -type f \( -name "*.mp4" -o -name "*.mov" \) 2>/dev/null | wc -l)
        if [ "$video_count" -gt 0 ]; then
            echo -e "  ${GREEN}✅ $video_count videos in data/raw/25_nov_2025/${NC}"
        else
            echo -e "  ${YELLOW}⚠️  No videos found in data/raw/25_nov_2025/${NC}"
            echo "     Add your test videos here to process them"
        fi
    else
        echo -e "  ${RED}❌ data/raw/25_nov_2025/ directory not found${NC}"
    fi
    
    echo ""
    
    # Outputs
    echo "📊 Results:"
    if [ -d "results" ]; then
        tracked_count=$(find results -name "*_tracked.mp4" 2>/dev/null | wc -l)
        csv_count=$(find results -name "*_detections.csv" 2>/dev/null | wc -l)
        json_count=$(find results -name "*_trajectory.json" 2>/dev/null | wc -l)
        
        if [ "$tracked_count" -gt 0 ] || [ "$csv_count" -gt 0 ]; then
            echo -e "  ${GREEN}✅ Tracked videos: $tracked_count${NC}"
            echo -e "  ${GREEN}✅ Detection CSVs: $csv_count${NC}"
            echo -e "  ${GREEN}✅ Trajectory JSONs: $json_count${NC}"
        else
            echo -e "  ${YELLOW}⚠️  No results yet${NC}"
            echo "     Run Option 3 to process videos"
        fi
    else
        echo -e "  ${YELLOW}⚠️  results/ directory not found${NC}"
    fi
    
    echo ""
    
    # Sample annotations
    echo "📝 Sample Annotations:"
    if [ -d "annotations" ]; then
        annotation_count=$(find annotations -name "*_detections.csv" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✅ $annotation_count sample CSV files${NC}"
    fi
    
    echo ""
}

# View outputs
view_outputs() {
    echo -e "${YELLOW}Output Results:${NC}"
    echo ""
    
    if [ -d "results" ] && [ "$(ls -A results 2>/dev/null)" ]; then
        echo "Location: $(pwd)/results/"
        echo ""
        
        # Tracked videos
        if ls results/*_tracked.mp4 1> /dev/null 2>&1; then
            echo "🎥 Tracked Videos:"
            ls -lh results/*_tracked.mp4 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
        
        # CSVs
        if ls results/*_detections.csv 1> /dev/null 2>&1; then
            echo "📊 Detection CSVs:"
            ls -lh results/*_detections.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
        
        # JSONs
        if ls results/*_trajectory.json 1> /dev/null 2>&1; then
            echo "📈 Trajectory JSONs:"
            ls -lh results/*_trajectory.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠️  No outputs yet.${NC}"
        echo "Run Option 3 to process videos and generate results."
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
            download_model
            ;;
        2)
            download_results
            ;;
        3)
            process_all
            ;;
        4)
            process_single
            ;;
        5)
            view_annotations
            ;;
        6)
            check_status
            ;;
        7)
            view_outputs
            ;;
        0)
            echo -e "${GREEN}Thank you for using the Cricket Ball Detection System!${NC}"
            echo "Good luck with your demo! 🎯"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Invalid option. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
    clear
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║     Cricket Ball Detection & Tracking System              ║"
    echo "║              Demo/Inference Mode                          ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
done