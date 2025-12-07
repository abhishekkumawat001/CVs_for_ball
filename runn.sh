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

# Get absolute path to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Google Drive IDs
MODEL_GDRIVE_ID="16SIU089-Eg6xTU8YcadvqNcBxxEs_Z_P"
VIDEOS_GDRIVE_ID="1A_R0eAUW1ZP4l866O3iBnv2rblYNvTQH"
MODEL_PATH="${SCRIPT_DIR}/runs/detect/local_finetune_optimized3/weights/best.pt"

# Function to show menu
show_menu() {
    echo "Select Operation:"
    echo ""
    echo "  SETUP (First Time):"
    echo "    1) Download trained model (best.pt) from Google Drive"
    echo "    2) Download sample results videos (optional)"
    echo ""
    echo "  INFERENCE (Main Demo):"
    echo "    3) Process videos from custom directory ⭐"
    echo "    4) Process single video (specify path)"
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
    mkdir -p "${SCRIPT_DIR}/runs/detect/local_finetune_optimized3/weights"
    
    # Download model
    echo "Downloading from Google Drive (5.4 MB)..."
    gdown ${MODEL_GDRIVE_ID} -O "${MODEL_PATH}"
    
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
    
    mkdir -p "${SCRIPT_DIR}/results"
    
    echo "Downloading from Google Drive..."
    gdown ${VIDEOS_GDRIVE_ID} -O "${SCRIPT_DIR}/results/edgefleet_results_videos.tar.gz"
    
    if [ -f "${SCRIPT_DIR}/results/edgefleet_results_videos.tar.gz" ]; then
        echo "Extracting archive..."
        tar -xzf "${SCRIPT_DIR}/results/edgefleet_results_videos.tar.gz" -C "${SCRIPT_DIR}/results"
        rm "${SCRIPT_DIR}/results/edgefleet_results_videos.tar.gz"
        
        # Fix directory structure if needed
        if [ -d "${SCRIPT_DIR}/results/results" ]; then
            mv "${SCRIPT_DIR}/results/results"/* "${SCRIPT_DIR}/results/"
            rm -r "${SCRIPT_DIR}/results/results"
        fi
        
        echo -e "${GREEN}✅ Sample results downloaded and extracted!${NC}"
        echo "Location: ${SCRIPT_DIR}/results/"
    else
        echo -e "${RED}❌ Download failed. Please download manually from:${NC}"
        echo "https://drive.google.com/file/d/${VIDEOS_GDRIVE_ID}/view"
    fi
}

# Process all videos from custom directory
process_all() {
    # Check if model exists
    if [ ! -f "${MODEL_PATH}" ]; then
        echo -e "${RED}❌ Model not found!${NC}"
        echo "Please download the model first (Option 1)"
        return 1
    fi
    
    echo ""
    echo -e "${YELLOW}Enter the directory containing your videos:${NC}"
    echo "Examples:"
    echo "  - Relative: ./videos or data/raw/test_videos"
    echo "  - Absolute: /home/user/Downloads/cricket_videos"
    echo ""
    echo -ne "Video directory path: "
    read video_dir
    
    # Expand tilde and resolve path
    video_dir="${video_dir/#\~/$HOME}"
    
    # Check if directory exists
    if [ ! -d "$video_dir" ]; then
        echo -e "${RED}❌ Directory not found: $video_dir${NC}"
        return 1
    fi
    
    # Count videos using find
    video_count=$(find "$video_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" \) 2>/dev/null | wc -l)
    
    if [ "$video_count" -eq 0 ]; then
        echo -e "${RED}❌ No video files found in: $video_dir${NC}"
        echo "Supported formats: .mp4, .mov, .avi"
        return 1
    fi
    
    echo -e "${GREEN}Found $video_count video(s)${NC}"
    echo ""
    
    # List videos
    echo -e "${YELLOW}Videos to process:${NC}"
    find "$video_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" \) | nl
    echo ""
    
    echo -ne "Process all these videos? (y/n): "
    read confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Cancelled."
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}Processing videos...${NC}"
    
    # Create output directory
    mkdir -p "${SCRIPT_DIR}/results"
    
    # Process each video
    processed=0
    while IFS= read -r video_file; do
        filename=$(basename "$video_file")
        echo -e "${YELLOW}[$((processed+1))/$video_count] Processing: $filename${NC}"
        
        cd "${SCRIPT_DIR}/code/inference"
        python inference_with_tracking.py --model "${MODEL_PATH}" --video "$video_file" --output-dir "${SCRIPT_DIR}/results/" 2>&1 | grep -v "^$"
        cd "${SCRIPT_DIR}"
        
        ((processed++))
    done < <(find "$video_dir" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.avi" \))
    
    echo ""
    echo -e "${GREEN}✅ Processed $processed video(s) successfully!${NC}"
    echo ""
    echo "Outputs saved to: ${SCRIPT_DIR}/results/"
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
    echo -e "${YELLOW}Enter the full path to your video file:${NC}"
    echo "Examples:"
    echo "  - Relative: ./video.mp4"
    echo "  - Absolute: /home/user/Downloads/cricket.mp4"
    echo ""
    echo -ne "Video file path: "
    read video_path
    
    # Expand tilde and resolve path
    video_path="${video_path/#\~/$HOME}"
    
    # Convert relative path to absolute
    if [[ ! "$video_path" = /* ]]; then
        video_path="${SCRIPT_DIR}/${video_path}"
    fi
    
    if [ ! -f "$video_path" ]; then
        echo -e "${RED}❌ Video file not found: $video_path${NC}"
        return 1
    fi
    
    filename=$(basename "$video_path")
    echo -e "${YELLOW}Processing: $filename${NC}"
    echo ""
    
    mkdir -p "${SCRIPT_DIR}/results"
    
    cd "${SCRIPT_DIR}/code/inference"
    python inference_with_tracking.py --model "${MODEL_PATH}" --video "$video_path" --output-dir "${SCRIPT_DIR}/results/"
    cd "${SCRIPT_DIR}"
    
    echo ""
    echo -e "${GREEN}✅ Video processed successfully!${NC}"
    echo "Output saved to: ${SCRIPT_DIR}/results/"
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
    
    if [ -f "${SCRIPT_DIR}/annotations/1_detections.csv" ]; then
        echo -e "${YELLOW}Example from annotations/1_detections.csv:${NC}"
        head -10 "${SCRIPT_DIR}/annotations/1_detections.csv"
        echo "..."
        echo ""
        total_frames=$(tail -n +2 "${SCRIPT_DIR}/annotations/1_detections.csv" | wc -l)
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
    
    # Outputs
    echo "📊 Results:"
    if [ -d "${SCRIPT_DIR}/results" ]; then
        tracked_count=$(find "${SCRIPT_DIR}/results" -name "*_tracked.mp4" 2>/dev/null | wc -l)
        csv_count=$(find "${SCRIPT_DIR}/results" -name "*_detections.csv" 2>/dev/null | wc -l)
        json_count=$(find "${SCRIPT_DIR}/results" -name "*_trajectory.json" 2>/dev/null | wc -l)
        
        if [ "$tracked_count" -gt 0 ] || [ "$csv_count" -gt 0 ]; then
            echo -e "  ${GREEN}✅ Tracked videos: $tracked_count${NC}"
            echo -e "  ${GREEN}✅ Detection CSVs: $csv_count${NC}"
            echo -e "  ${GREEN}✅ Trajectory JSONs: $json_count${NC}"
        else
            echo -e "  ${YELLOW}⚠️  No results yet${NC}"
            echo "     Run Option 3 or 4 to process videos"
        fi
    else
        echo -e "  ${YELLOW}⚠️  results/ directory not found${NC}"
    fi
    
    echo ""
    
    # Sample annotations
    echo "📝 Sample Annotations:"
    if [ -d "${SCRIPT_DIR}/annotations" ]; then
        annotation_count=$(find "${SCRIPT_DIR}/annotations" -name "*_detections.csv" 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✅ $annotation_count sample CSV files${NC}"
    fi
    
    echo ""
}

# View outputs
view_outputs() {
    echo -e "${YELLOW}Output Results:${NC}"
    echo ""
    
    if [ -d "${SCRIPT_DIR}/results" ] && [ "$(ls -A ${SCRIPT_DIR}/results 2>/dev/null)" ]; then
        echo "Location: ${SCRIPT_DIR}/results/"
        echo ""
        
        # Tracked videos
        if ls "${SCRIPT_DIR}/results"/*_tracked.mp4 1> /dev/null 2>&1; then
            echo "🎥 Tracked Videos:"
            ls -lh "${SCRIPT_DIR}/results"/*_tracked.mp4 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
        
        # CSVs
        if ls "${SCRIPT_DIR}/results"/*_detections.csv 1> /dev/null 2>&1; then
            echo "📊 Detection CSVs:"
            ls -lh "${SCRIPT_DIR}/results"/*_detections.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
        
        # JSONs
        if ls "${SCRIPT_DIR}/results"/*_trajectory.json 1> /dev/null 2>&1; then
            echo "📈 Trajectory JSONs:"
            ls -lh "${SCRIPT_DIR}/results"/*_trajectory.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
            echo ""
        fi
    else
        echo -e "${YELLOW}⚠️  No outputs yet.${NC}"
        echo "Run Option 3 or 4 to process videos and generate results."
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
