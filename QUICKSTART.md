# 🚀 QUICKSTART GUIDE


### 1. Setup (One Time)
```bash
./setup.sh
```

### 2. Run Everything
```bash
./run.sh
```
Then select **Option 4** to process all videos.

---

## That's It! 🎉

### Outputs
All results will be in: `output/tracked_videos/`
- 15 tracked videos with trajectories
- 15 CSV files with frame-by-frame data
- 15 trajectory JSON files with analytics

---

## Common Tasks

### Process Videos (Most Common)
```bash
./run.sh → Select 4
```

### Train New Model
```bash
./run.sh → Select 1
```
(Takes ~45 minutes on GPU)

### Check Status
```bash
./run.sh → Select 6
```

### View Outputs
```bash
./run.sh → Select 7
```

---

## File Structure (Simplified)

```
edgefleet/
├── run.sh              ← 🎯 START HERE (main launcher)
├── setup.sh            ← Run once to install dependencies
│
├── code/            ← All scripts organized here
│   ├── training/       ← Training scripts
│   └── inference/      ← Detection & tracking scripts
│
├── models/weights/     ← YOLO models
├── data/raw/           ← Input videos
└── output/             ← Results go here
```

---

## Need Help?

**Check status**: `./run.sh` → Option 6  
**View README**: `cat README.md`  
**Check outputs**: `ls output/tracked_videos/`

---

## Quick Reference

| Task | Command |
|------|---------|
| Process all videos | `./run.sh` → 4 |
| Process one video | `./run.sh` → 5 |
| Train model | `./run.sh` → 1 |
| Check status | `./run.sh` → 6 |
| View outputs | `./run.sh` → 7 |

**That's all you need to know!** 🎯
