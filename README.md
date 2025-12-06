# Cricket Ball Detection & Tracking System

A production-ready computer vision system for cricket ball detection and tracking using YOLO11n + Kalman filtering. Built as part of the EdgeFleet AI Test Kit for IISc.

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-11n-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📦 Required Downloads

### Model Weights (Required - 5.4MB)
**Download trained YOLO11n model** to run inference:

📥 [**Download best.pt from Google Drive**](https://drive.google.com/file/d/16SIU089-Eg6xTU8YcadvqNcBxxEs_Z_P/view?usp=sharing)

**Installation:**
```bash
# After downloading best.pt, place it in the correct location:
mkdir -p runs/detect/local_finetune_optimized3/weights
mv ~/Downloads/best.pt runs/detect/local_finetune_optimized3/weights/
```

Or download directly using `gdown`:
```bash
pip install gdown
gdown 16SIU089-Eg6xTU8YcadvqNcBxxEs_Z_P
mkdir -p runs/detect/local_finetune_optimized3/weights
mv best.pt runs/detect/local_finetune_optimized3/weights/
```

### Processed Videos (Optional - 123MB)
**Example outputs** - 15 videos with trajectory overlays:

📥 [**Download edgefleet_results_videos.tar.gz from Google Drive**](https://drive.google.com/file/d/1A_R0eAUW1ZP4l866O3iBnv2rblYNvTQH/view?usp=sharing)

**Extract:**
```bash
# After downloading the archive
tar -xzf edgefleet_results_videos.tar.gz
# Videos will be in results/ directory
```

**Note:** You can generate your own results by running the inference pipeline (see Quick Start below).

---

## 🚀 Quick Start

### Step 1: Setup (First Time Only)
```bash
# Install dependencies
./setup.sh
```

### Step 2: Run the System
```bash
# Launch interactive menu
./run.sh
```

**Simple Menu Options:**
- **Option 4**: Process all videos (tracked + CSVs) ⭐ **MOST COMMON**
- **Option 1**: Train complete pipeline (if starting from scratch)
- **Option 6**: Check status and outputs

### Alternative: Direct Commands
```bash
# Process all test videos (most common use case)
cd code/inference && ./generate_all_csvs.sh

# Train from scratch
cd code/training && ./train_kaggle_pretrain.sh && ./train_local_finetune.sh

# Check status
cd code/inference && ls ../output/tracked_videos/
```

## 📁 Repository Structure

```
edgefleet/
├── run.sh                           # 🚀 Main launcher (USE THIS)
├── setup.sh                         # One-time setup script
│
├── code/
│   ├── training/                    # Model training scripts
│   │   ├── train_kaggle_pretrain.sh
│   │   ├── train_local_finetune.sh
│   │
│   ├── inference/                   # Detection & tracking scripts
│   │   ├── inference_with_tracking.py
│   │   ├── predict_optimized.sh
│   │   ├── process_all_videos.sh
│   │   ├── generate_all_csvs.sh
│   │   └── run_tracking.sh
│   │
│   └── utils/                       # Configuration files
│       ├── kaggle_dataset.yaml
│       └── local_dataset.yaml
│
├── src/                             # Core source code
│   ├── detection/                   # Ball detection modules
│   ├── tracking/                    # Tracking algorithms (Kalman, Optical Flow)
│   ├── preprocessing/               # Video/image preprocessing
│   ├── postprocessing/              # Visualization and output
│   └── utils/                       # Utility functions
│
├── data/
│   ├── raw/25_nov_2025/            # 15 test videos
│   └── processed/
│
├── output/
│   ├── tracked_videos/              # Tracked videos + CSVs + JSONs
│   ├── frames/
│   └── trajectories/
│
├── runs/detect/                     # Training runs
│   ├── kaggle_pretrain_optimized2/
│   └── local_finetune_optimized3/   # Production model
│
├── models/
│   └── weights/
│       └── yolo11n.pt               # Base YOLO11n model
│
├── docs/                            # Documentation & PDFs
├── dataset_from kaggle/             # Kaggle dataset
└── dataset_local/                   # Local finetuning dataset
```

---

## 📊 Complete Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────┐
    │  train_kaggle_pretrain.sh                    │
    │  • Dataset: Kaggle (1778 train, 63 val)     │
    │  • Model: YOLO11n (2.59M params)            │
    │  • Epochs: 50, Image size: 640px            │
    │  • Output: kaggle_pretrain_optimized2/      │
    │  • Time: ~5-10 minutes                       │
    └──────────────────────────────────────────────┘
                            │
                            ▼
    ┌──────────────────────────────────────────────┐
    │  train_local_finetune.sh                     │
    │  • Dataset: Local (152 train, 26 val)       │
    │  • Transfer from: Kaggle weights             │
    │  • Epochs: 300, Image size: 1280px          │
    │  • Learning rate: 0.0005 (reduced)          │
    │  • Output: local_finetune_optimized3/       │
    │  • Time: ~30-40 minutes                      │
    └──────────────────────────────────────────────┘
                            │
                            ▼
        Production Model: best.pt (53.2% mAP@0.5)

┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PIPELINE                        │
└─────────────────────────────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
    ┌───────────────────┐   ┌──────────────────┐
    │ Single Video      │   │ Batch Processing │
    │ predict_optimized │   │ generate_all_csvs│
    └───────────────────┘   └──────────────────┘
                    │               │
                    └───────┬───────┘
                            ▼
            ┌──────────────────────────────┐
            │ inference_with_tracking.py   │
            │  • YOLO Detection            │
            │  • Kalman Tracking           │
            │  • Trajectory Analysis       │
            │  • CSV Generation            │
            └──────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │ Outputs (output/tracked_videos/)      │
        │  • *_tracked.mp4 (with trajectories)  │
        │  • *_detections.csv (frame data)      │
        │  • *_trajectory.json (analytics)      │
        └───────────────────────────────────────┘
```

---

## 🎯 Training Scripts

### 1. train_kaggle_pretrain.sh
**Purpose**: Pretrain YOLO11n model on Kaggle cricket ball dataset

**Usage**:
```bash
cd code/training
./train_kaggle_pretrain.sh
```

**Parameters**:
- Model: YOLO11n (2.59M parameters)
- Dataset: Kaggle (1778 train, 63 val, 71 test)
- Epochs: 50
- Image size: 640px
- Batch: 16
- Output: `runs/detect/kaggle_pretrain_optimized2/weights/best.pt`

---

### 2. train_local_finetune.sh
**Purpose**: Finetune on local dataset using pretrained Kaggle weights

**Usage**:
```bash
cd code/training
./train_local_finetune.sh
```

**Parameters**:
- Model: Pretrained from Kaggle (best.pt)
- Dataset: Local (152 train, 26 val, 52 test)
- Epochs: 300
- Image size: 1280px (2x for small objects)
- Batch: 16
- Learning rate: 0.0005 (reduced for finetuning)
- Output: `runs/detect/local_finetune_optimized3/weights/best.pt` ⭐ **PRODUCTION MODEL**

**Training Pipeline**:
```
Step 1: train_kaggle_pretrain.sh
   ↓ (generates kaggle_pretrain_optimized2/weights/best.pt)
Step 2: train_local_finetune.sh
   ↓ (generates local_finetune_optimized3/weights/best.pt)
```

**Notes**:
- Always run kaggle pretrain before local finetuning
- Local finetuning requires: `runs/detect/kaggle_pretrain_optimized2/weights/best.pt`
- Best production model: `local_finetune_optimized3/weights/best.pt`

---

## 🎬 Inference Scripts

### 1. predict_optimized.sh
**Purpose**: Run YOLO detection on single video with optimized parameters

**Usage**:
```bash
cd code/inference
./predict_optimized.sh [model_path] [video_path]

# Examples:
./predict_optimized.sh "" ../../data/raw/25_nov_2025/1.mp4
./predict_optimized.sh ../../runs/detect/local_finetune_optimized3/weights/best.pt ../../data/raw/25_nov_2025/2.mov
```

**Default Model**: `runs/detect/local_finetune_optimized3/weights/best.pt`

**Parameters**:
- Confidence: 0.1
- Image size: 1280px
- Max detections: 5
- Output: `runs/detect/predict_optimized/`

---

### 2. inference_with_tracking.py
**Purpose**: Core inference + Kalman tracking + trajectory analysis + CSV generation

**Usage**:
```bash
cd code/inference

# Single video
python inference_with_tracking.py --video ../../data/raw/25_nov_2025/1.mp4

# All videos in directory
python inference_with_tracking.py --input-dir ../../data/raw/25_nov_2025

# Custom parameters
python inference_with_tracking.py \
    --model ../../runs/detect/local_finetune_optimized3/weights/best.pt \
    --video ../../data/raw/25_nov_2025/1.mp4 \
    --conf 0.15 \
    --imgsz 1280 \
    --output-dir ../../output/custom/
```

**Outputs**:
- Tracked videos: `*_tracked.mp4` (with trajectory trails)
- CSV files: `*_detections.csv` (per-frame data)
- JSON files: `*_trajectory.json` (analysis metrics)

**Parameters**:
- `--model`: Model path (default: optimized3/best.pt)
- `--conf`: Confidence threshold (default: 0.1)
- `--imgsz`: Image size (default: 1280)
- `--output-dir`: Output directory (default: output/tracked_videos)

---

### 3. process_all_videos.sh
**Purpose**: Batch process all 15 videos with YOLO detection only (no tracking)

**Usage**:
```bash
cd code/inference
./process_all_videos.sh
```

**Output**: `runs/detect/predict_optimized/*.avi`

---

### 4. generate_all_csvs.sh
**Purpose**: Generate CSVs + tracked videos for all 15 test videos

**Usage**:
```bash
cd code/inference
./generate_all_csvs.sh
```

**Output**: `output/tracked_videos/`
- 15 tracked videos (`*_tracked.mp4`)
- 15 CSV files (`*_detections.csv`)
- 15 trajectory JSONs (`*_trajectory.json`)

---

### 5. run_tracking.sh
**Purpose**: Quick wrapper for inference_with_tracking.py

**Usage**:
```bash
cd code/inference
./run_tracking.sh
```

---

## 📦 Dataset Configuration (code/utils/)

### kaggle_dataset.yaml
YOLO dataset configuration for Kaggle cricket ball dataset.

- **Path**: `dataset_from kaggle/`
- **Classes**: 1 (ball)
- **Split**: 1778 train, 63 val, 71 test
- **Used by**: `code/training/train_kaggle_pretrain.sh`

---

### local_dataset.yaml
YOLO dataset configuration for local cricket dataset.

- **Path**: `dataset_local/`
- **Classes**: 1 (ball)
- **Split**: 152 train, 26 val, 52 test (video-level split)
- **Used by**: `code/training/train_local_finetune.sh`

---

## 📄 Output Formats

### CSV Format (per-frame data):
```csv
frame,timestamp,detections,tracked_objects,ball_1_x,ball_1_y,ball_1_confidence,ball_1_track_id
1,0.04,3,1,881.0,351.0,0.497,0.0
2,0.08,3,1,881.0,350.0,0.349,0.0
3,0.12,3,1,881.0,349.0,0.428,0.0
```

**Columns**:
- `frame`: Frame number
- `timestamp`: Video timestamp (seconds)
- `detections`: Total YOLO detections in frame
- `tracked_objects`: Number of tracked objects (Kalman filter)
- `ball_N_x/y`: Ball position coordinates
- `ball_N_confidence`: YOLO confidence score
- `ball_N_track_id`: Kalman tracker ID

---

### Tracked Videos:
- **Trajectory trails**: Magenta lines (last 30 frames)
- **Bounding boxes**: Green rectangles
- **Track IDs + Confidence**: Text overlays
- **Frame counters**: Top-left corner

---

### Trajectory JSON:
```json
{
  "video": "1.mp4",
  "total_frames": 31,
  "detected_frames": 18,
  "detection_rate": 0.58,
  "avg_speed_pixels_per_frame": 57.8,
  "max_speed": 124.3,
  "total_distance": 1041.2,
  "bounces": 0
}
```

---

## 📈 Model Performance

**Production Model**: `local_finetune_optimized3/weights/best.pt`
- **mAP@0.5**: 53.2%
- **Precision**: 60.9%
- **Recall**: 61.5%
- **Detection Rate**: ~67% across 15 test videos (196/292 frames)
- **Parameters**: 2.59M
- **Architecture**: YOLO11n

---

## 🛠️ Scripts Overview

### Training Scripts
| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `train_kaggle_pretrain.sh` | Pretrain on Kaggle | ~5-10 min | kaggle_pretrain_optimized2/weights/best.pt |
| `train_local_finetune.sh` | Finetune on local data | ~30-40 min | local_finetune_optimized3/weights/best.pt |

### Inference Scripts
| Script | Purpose | Time | Output |
|--------|---------|------|--------|
| `predict_optimized.sh` | Single video detection | <1 min | predict_optimized/*.avi |
| `process_all_videos.sh` | Batch detection (15 videos) | ~3 min | predict_optimized/*.avi |
| `generate_all_csvs.sh` | Full tracking pipeline | ~3-4 min | tracked_videos/*.mp4/csv/json |
| `inference_with_tracking.py` | Core tracking engine | - | Flexible output |

---

## 💡 Usage Examples

### Complete Training from Scratch
```bash
./run.sh
# Select: 1) Train complete pipeline

# Or manually:
cd code/training
./train_kaggle_pretrain.sh && ./train_local_finetune.sh
```

### Process All Videos with Tracking (MOST COMMON)
```bash
./run.sh
# Select: 4) Process all videos

# Or directly:
cd code/inference && ./generate_all_csvs.sh
```

### Process Single Video
```bash
./run.sh
# Select: 5) Process single video

# Or directly:
cd code/inference
python inference_with_tracking.py --video ../../data/raw/25_nov_2025/1.mp4
```

### Custom Inference Parameters
```bash
cd code/inference
python inference_with_tracking.py \
    --model ../../runs/detect/local_finetune_optimized3/weights/best.pt \
    --video ../../data/raw/25_nov_2025/1.mp4 \
    --conf 0.15 \
    --output-dir ../../output/custom/
```

---

## ⚙️ Configuration

All paths are relative to the repository root:
- **Models**: `runs/detect/*/weights/best.pt`
- **Datasets**: `dataset_from kaggle/`, `dataset_local/`
- **Test videos**: `data/raw/25_nov_2025/` (15 videos)
- **Outputs**: `output/tracked_videos/`

**Important**: All scripts must be run from their respective directories to ensure correct path resolution.

---

## 🔧 Requirements

- Python 3.10+
- CUDA 12.8
- Conda environment: `swe`
- GPU: 3x NVIDIA RTX A6000 (47.5GB VRAM each) - recommended for training
- Packages: `ultralytics`, `opencv-python`, `numpy`, `scipy`

---

## 📝 Notes

- All scripts use production model: `local_finetune_optimized3/weights/best.pt`
- Default confidence threshold: 0.1 (adjustable with `--conf` parameter)
- Training requires GPU (3x RTX A6000 recommended)
- Total training time: ~40-50 minutes
- Inference time: ~3-4 minutes for 15 videos
- Conda environment `swe` must be activated
- Video-level dataset splitting prevents frame leakage

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 📧 Contact

For questions and support:
- Email: support@edgefleet.ai
- Issues: GitHub Issues

---

## 🙏 Acknowledgments

- EdgeFleet AI Test Kit
- Indian Institute of Science (IISc)
- Ultralytics YOLO Community
- OpenCV Community
