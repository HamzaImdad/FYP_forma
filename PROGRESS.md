# ExerVision - Progress Log

## Day 1 (23 Feb 2026) - Data Collection & Setup

### Completed
1. **Python Environment** - Python 3.10.6 confirmed at `C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe`
2. **Dependencies Installed** - mediapipe, opencv-python, numpy, pandas, yt-dlp, kaggle, tqdm
3. **Project Directory Structure** - Created full data directory tree for all 10 exercises
4. **Kaggle API** - Connected (username: hamzaimdaderfef)
5. **Kaggle Workout Images Downloaded** (818MB) - 22 exercise categories, ~6,000+ images
   - squat: 742, deadlift: 530, bench press: 625, shoulder press: 512, pull up: 615
   - push up: 601, plank: 993, bicep curl: 705, tricep dips: 698
6. **Kaggle Workout Videos Downloaded** (4.32GB) - 22 exercise categories, 310+ videos
   - squat: 29, deadlift: 32, bench press: 61, shoulder press: 17, pull up: 26
   - push-up: 56, plank: 7, bicep curl: 62, tricep dips: 20
7. **YouTube Videos Downloaded** (114 videos) - Search-based download for all 10 exercises
   - squat: 8 correct / 11 incorrect
   - deadlift: 7 correct / 5 incorrect
   - bench_press: 8 correct / 3 incorrect
   - overhead_press: 8 correct / 6 incorrect
   - lunge: 5 correct / 5 incorrect
   - pushup: 4 correct / 5 incorrect
   - pullup: 6 correct / 3 incorrect
   - plank: 6 correct / 3 incorrect
   - bicep_curl: 4 correct / 5 incorrect
   - tricep_dip: 8 correct / 4 incorrect
8. **MediaPipe Model Downloaded** - pose_landmarker_heavy.task
9. **Landmark Extraction from Kaggle Images** - ~5,600 correct-form samples extracted to CSV
   - 9 CSV files in `data/processed/landmarks/`
   - Detection rates: 77-99% across exercises
   - Each CSV has 33 landmarks × 7 values (x, y, z, vis, wx, wy, wz) per row

### Scripts Created
- `scripts/download_youtube.py` - v1 (hardcoded URLs, mostly failed)
- `scripts/download_youtube_search.py` - v2 (search-based, unused)
- `scripts/download_youtube_v3.py` - v3 (search-based, working - used for final download)
- `scripts/extract_landmarks.py` - Batch video landmark extraction (for YouTube + Kaggle videos)
- `scripts/extract_landmarks_images.py` - Batch image landmark extraction (used for Kaggle images)

### Data Location Summary
```
data/
  datasets/
    kaggle_images/       - 818MB of exercise images (22 categories)
    kaggle_workout/      - 4.32GB of exercise videos (22 categories)
    youtube/             - 114 videos organized by exercise/label
  processed/
    landmarks/           - 9 CSV files (~25MB total) with extracted pose landmarks
  raw/                   - Empty (for self-recorded videos)
```

---

## TODO for Day 2

### Priority 1: Extract Remaining Landmarks
- [ ] Extract landmarks from YouTube videos (114 videos, both correct AND incorrect form)
  - Command: `python scripts/extract_landmarks.py --source youtube`
  - This gives us the crucial INCORRECT form data
- [ ] Extract landmarks from Kaggle workout videos (310+ videos)
  - Command: `python scripts/extract_landmarks.py --source kaggle`

### Priority 2: Feature Engineering
- [ ] Build feature computation script (joint angles, ROM, velocity, symmetry)
- [ ] Compute features from all landmark CSVs
- [ ] Create combined dataset with exercise type + correct/incorrect labels

### Priority 3: Data Augmentation & Splits
- [ ] Apply augmentation (noise, rotation, scaling, flipping)
- [ ] Create stratified train/val/test splits (70/15/15)

### Priority 4: Start Building Core Application
- [ ] Project scaffolding (constants, config, utils)
- [ ] Pose estimation engine (base classes + MediaPipe wrapper)
- [ ] Rule-based classifiers for first 3 exercises

---

## Datasets Not Yet Downloaded (Optional)
- **REHAB24-6** (Zenodo) - Download failed (redirect issue). URL: https://zenodo.org/records/13305826
- **InfiniteRep** (synthetic) - Not downloaded yet. URL: https://marketplace.infinity.ai/pages/infiniterep-dataset
- **Fitness-AQA** - Requires access request: https://forms.gle/PbPTX1eVxGpa3QG88
- **FLEX Dataset** - Requires access request: https://haoyin116.github.io/FLEX_Dataset/

## GitHub
- Repo: https://github.com/HamzaImdad/FYP.git
- Status: NOT YET INITIALIZED - will set up git and start committing incrementally
