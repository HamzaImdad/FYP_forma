# ExerVision - CLAUDE.md

## Frontend Design Skills (MUST USE)

The following design skills are installed at `.claude/commands/skills/` and MUST be used when working on any UI/frontend task:

- **frontend-design** — ALWAYS use this skill for ANY UI work. Creates distinctive, production-grade interfaces. Avoid generic AI aesthetics. Bold typography, unique fonts, intentional color, motion, spatial composition, textures.
- **modern-web-design** — Modern design trends 2024-2025: performance-first, bold minimalism, micro-interactions, scrollytelling, glassmorphism, accessibility.
- **animation-patterns** — CSS animation patterns: entrance reveals, background effects (gradient mesh, noise, grid), interactive 3D tilt, troubleshooting.
- **micro-interactions** — Subtle hover effects, loading states, form feedback, transition animations. Performance-optimized, accessibility-aware.
- **ui-design-methodology** — Systematic UI/UX design: semantic tokens, color psychology, spacing scales, component variants, quality checklists.
- **scroll-reveal-libraries** — AOS (Animate On Scroll): data-attribute API, 50+ animations, framework-agnostic.
- **lightweight-3d-effects** — Vanta.js backgrounds, Vanilla-Tilt parallax cards, Zdog pseudo-3D illustrations.
- **animated-component-libraries** — Magic UI + React Bits: pre-built animated components, backgrounds, text effects.
- **gsap-scrolltrigger** — GSAP ScrollTrigger: scroll-driven animations, pinning, scrubbing, parallax.
- **animejs** — Anime.js: timeline animations, stagger effects, SVG morphing.
- **locomotive-scroll** — Smooth scrolling with parallax and viewport detection.
- **barba-js** — Page transitions between routes.
- **mobile-design** — Mobile-specific design patterns and responsive strategies.

### Design Direction
- **Aesthetic**: Industrial luxury fitness — dark charcoal (#0D0D0D), warm gold/copper accent (#D4A574), burnt orange primary (#E8572A)
- **Typography**: Bebas Neue (display, condensed, powerful) + Outfit (body) + Cormorant Garamond (italic accents)
- **Images**: Every section must be visually rich — exercise cards have real photos, sections use full-bleed backgrounds
- **Gold throughout**: Logo, accents, dividers, em tags — warm gold (#D4A574) is the signature color

## Project Overview

ExerVision is a **real-time computer vision exercise form evaluator** built as a BSc CS Final Year Project at the University of Greenwich.

- Uses **MediaPipe BlazePose** (33 landmarks) for pose estimation
- Supports **10 exercises**: squat, lunge, deadlift, bench_press, overhead_press, pullup, pushup, plank, bicep_curl, tricep_dip
- **3 classifier modes**: rule_based, ml (Random Forest/SVM/LR), bilstm (CNN-BiLSTM + attention)
- **Web interface**: Flask + Socket.IO, single-page app with real-time webcam processing
- Continuous form scoring (0-100) with color-coded per-joint feedback

## Environment

- **Python**: 3.10.6 at `C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe`
- `python` is NOT on PATH — use full path or `pip` directly
- **GPU**: NVIDIA GPU available (CUDA)
- **OS**: Windows 11, shell is Git Bash (use Unix paths in commands)
- **Project path has spaces**: `"/c/Users/Hamza/OneDrive - University of Greenwich/Documents/fyp github"` — always quote paths

## Project Structure

```
root/
├── app/                          # Web application
│   ├── server.py                 # Flask + Socket.IO server — emits form_score + is_active
│   ├── exercise_data.json        # Exercise metadata (muscles, instructions, mistakes)
│   ├── static/
│   │   ├── app.js                # Client-side SPA — live score display, score history chart
│   │   ├── style.css             # UI styles — score coloring (red/orange/green)
│   │   └── images/               # Feature showcase images
│   └── templates/
│       └── index.html            # Single HTML template — score display elements
│
├── src/                          # Core library (imported by app and scripts)
│   ├── pose_estimation/
│   │   ├── base.py               # PoseResult dataclass (landmarks: 33x4, world_landmarks: 33x3)
│   │   └── mediapipe_estimator.py # MediaPipe wrapper (IMAGE and VIDEO modes)
│   ├── feature_extraction/
│   │   ├── landmark_features.py  # LandmarkFeatureExtractor: raw 99-dim normalized landmarks
│   │   ├── angles.py             # compute_joint_angles() from PoseResult
│   │   ├── exercise_features.py  # EXERCISE_FEATURES dict + use_landmarks flag
│   │   ├── features.py           # FeatureExtractor: angles + custom + temporal (legacy)
│   │   ├── rep_counter.py        # RepCounter: peak detection on angle time series
│   │   └── temporal_features.py  # velocity, acceleration, ROM over frame windows
│   ├── classification/
│   │   ├── base.py               # ClassificationResult (now has is_active + form_score)
│   │   ├── rule_based.py         # RuleBasedClassifier: motion-based activity gate (10-frame buffer)
│   │   ├── ml_classifier.py      # MLClassifier: loads .pkl sklearn models per exercise
│   │   ├── bilstm_classifier.py  # CNN-BiLSTM: Conv1D front-end + BiLSTM + attention, hidden=128
│   │   ├── base_detector.py      # BaseExerciseDetector: shared state machine, rep counting, scoring
│   │   ├── squat_detector.py     # SquatDetector: knee angle + depth + torso lean + valgus
│   │   ├── deadlift_detector.py  # DeadliftDetector: hip hinge + back straightness
│   │   ├── lunge_detector.py     # LungeDetector: front/back knee depth + torso
│   │   ├── bench_press_detector.py # BenchPressDetector: elbow depth + symmetry + flare
│   │   ├── overhead_press_detector.py # OverheadPressDetector: lockout + torso lean
│   │   ├── pullup_detector.py    # PullUpDetector: chin above bar + body swing
│   │   ├── pushup_detector.py    # PushUpDetector: hip sag + depth + shoulder flare
│   │   ├── plank_detector.py     # PlankDetector: body line + hip sag (static hold)
│   │   ├── bicep_curl_detector.py # BicepCurlDetector: upper arm stability + torso swing
│   │   └── tricep_dip_detector.py # TricepDipDetector: elbow depth + forward lean
│   ├── feedback/
│   │   └── feedback_engine.py    # Score-based text (Good/Moderate/Needs improvement) + gradient colors
│   ├── visualization/
│   │   └── overlay.py            # draw_skeleton, draw_angle_zones, draw_feedback_panel
│   ├── pipeline/
│   │   └── realtime.py           # ExerVisionPipeline: supports landmark features, form_score tracking
│   ├── evaluation/
│   │   └── metrics.py            # evaluate_classifier, per_exercise_report, compare_models
│   └── utils/
│       ├── config.py             # Config dataclass (paths, MediaPipe settings)
│       ├── constants.py          # EXERCISES, LANDMARK_NAMES/INDICES, JOINT_ANGLES, SKELETON_CONNECTIONS
│       ├── geometry.py           # calculate_angle, distance, midpoint, vertical_angle
│       ├── temporal.py           # TemporalSmoother (rolling window low-pass filter)
│       └── augmentation.py       # Augmentation: pose-level + feature-level + landmark-level
│
├── scripts/
│   ├── train_bilstm.py           # CNN-BiLSTM training — warmup, cosine LR, val F1 stopping, --use-landmarks
│   ├── train_classifier.py       # Train ML classifiers (RF, SVM, LR) per exercise
│   ├── balance_data.py           # Clean + balance dataset (undersample majority, augment minority)
│   ├── build_features.py         # landmarks CSV -> features CSV (--landmarks for raw 99-dim)
│   ├── build_sequences.py        # features -> sliding window .npz (--use-landmarks for 99-dim)
│   ├── augment_data.py           # Data augmentation (--mode landmarks|features)
│   ├── extract_landmarks.py      # Video -> landmark CSV extraction
│   ├── extract_landmarks_images.py # Image -> landmark CSV extraction
│   ├── evaluate_models.py        # Evaluate single exercise
│   ├── evaluate_all.py           # Evaluate all exercises
│   └── benchmark_fps.py          # FPS profiling
│
├── models/
│   ├── mediapipe/                # pose_landmarker_lite.task, pose_landmarker_heavy.task
│   └── trained/                  # *_classifier.pkl, *_bilstm.pt, *_bilstm_v2.pt, *_eval.json
│
├── data/
│   ├── datasets/                 # Raw data (kaggle_images, kaggle_workout, youtube)
│   └── processed/
│       ├── landmarks/            # Extracted pose landmarks (CSVs)
│       ├── features/             # all_features.csv, landmark_features.csv, balanced_features.csv
│       └── sequences/            # *_sequences.npz + *_landmark_sequences.npz
│
└── reports/                      # Evaluation reports, training curves, confusion matrices
```

## Key Data Classes

```python
# src/pose_estimation/base.py
PoseResult:
    landmarks: np.ndarray       # shape (33, 4): x, y, z, visibility (image-normalized 0-1)
    world_landmarks: np.ndarray # shape (33, 3): x, y, z (meters, hip-centered)
    detection_confidence: float
    timestamp_ms: int

# src/classification/base.py
ClassificationResult:
    exercise: str
    is_correct: bool
    confidence: float           # 0.0-1.0
    joint_feedback: Dict[str, str]  # joint_name -> "correct"|"incorrect"|"warning"
    details: Dict[str, str]     # feature_name -> description of violation
    is_active: bool = True      # whether user is actually exercising (motion-based)
    form_score: float = 0.0     # continuous 0.0-1.0 form quality score
```

## Feature Extraction Modes

### Legacy (hand-crafted): `FeatureExtractor` in `features.py`
- 10-16 features per exercise (angles, ratios, symmetry, temporal)
- Used by: rule-based classifier, ML classifier, old BiLSTM

### New (raw landmarks): `LandmarkFeatureExtractor` in `landmark_features.py`
- 99 features per frame (33 landmarks x 3 coords)
- Body-centered normalization (hip midpoint origin, torso-length scaled)
- Visibility masking for occluded landmarks
- Used by: new CNN-BiLSTM (v2 models)
- `extract_full()` returns both raw landmarks + angle features together

## Data Pipeline

### Unified Pipeline (recommended)
```
python scripts/pipeline.py                          # Full pipeline, all exercises
python scripts/pipeline.py --exercise squat lunge   # Specific exercises only
python scripts/pipeline.py --skip-extract           # Skip extraction, rebuild+train
python scripts/pipeline.py --dry-run                # Show plan, don't execute
```

### Manual Steps
```
Raw videos → data/datasets/youtube/{exercise}/{correct|incorrect}/
  → extract_landmarks_parallel.py → data/processed/landmarks/ CSVs
  → split_videos.py --force → data/processed/splits/ manifests (train/val/test by video)
  → build_sequences.py --use-landmarks → data/processed/sequences/ .npz
  → train_bilstm.py --use-landmarks → models/trained/*_bilstm_v2.pt
```

### Per-Exercise Feature Dimensions (source of truth)
| Dim | Mode | Exercises |
|-----|------|-----------|
| 99 | landmarks | squat, deadlift, bench_press, bicep_curl |
| 109 | hybrid (99 lm + 10 angles) | pullup, pushup, plank |
| 330 | temporal (99 pos + 99 vel + 99 acc + 33 sym) | lunge, overhead_press, tricep_dip |

## Current State (as of 2026-04-07)

### What's DONE and working:
- **Dedicated exercise detectors for ALL 10 exercises** — state machine + angle thresholds ✓
  - BaseExerciseDetector base class with shared logic ✓
  - 9 new detectors (squat, deadlift, lunge, bench_press, overhead_press, pullup, plank, bicep_curl, tricep_dip) ✓
  - PushUpDetector (original, standalone) ✓
  - PlankDetector (static hold, tracks duration instead of reps) ✓
- All 10 exercises trained with CNN-BiLSTM v2 (cherry-picked best per exercise) ✓
- Unified data pipeline script (`scripts/pipeline.py`) ✓
- Video-level train/val/test splits (no data leakage) ✓
- Web app tested — all 10 BiLSTM models load and serve correctly ✓
- Session HUD with phase, progress, set tracking for ALL exercises ✓
- Activity gate (10-frame motion detection) ✓
- Continuous form scoring (0-100) with confidence gating ✓
- Web UI with live score display + score history chart ✓
- Session recording (480p WebM, 10fps, 500kbps) ✓
- Camera setup instructions for all 10 exercises ✓
- Model backup before overwriting ✓
- **All work committed to git** ✓

### Architecture: Dedicated Detectors (PRIMARY classification path)
All 10 exercises now use dedicated detectors that bypass ML models entirely:
- Each detector: angle computation → state machine → form scoring → joint feedback
- State machine: TOP → GOING_DOWN → BOTTOM → GOING_UP → TOP (1 rep)
- Per-frame form assessment with weighted scoring (exercise-specific)
- Set tracking with 8-second rest timeout
- Session summary with per-rep scores, common issues, set breakdown

| Exercise | Detector | Primary Angle | Key Form Checks |
|---|---|---|---|
| squat | SquatDetector | knee (hip-knee-ankle) | depth, torso lean, knee valgus |
| deadlift | DeadliftDetector | hip (shoulder-hip-knee) | back straightness, knee bend |
| lunge | LungeDetector | front knee | front/back knee depth, torso upright |
| bench_press | BenchPressDetector | elbow | depth, symmetry, shoulder flare |
| overhead_press | OverheadPressDetector | elbow | lockout, torso lean/back arch |
| pullup | PullUpDetector | elbow | chin above bar, body swing |
| pushup | PushUpDetector | elbow | hip sag, depth, shoulder flare |
| plank | PlankDetector (static) | body line | hip sag/pike, shoulder alignment |
| bicep_curl | BicepCurlDetector | elbow | upper arm stability, torso swing |
| tricep_dip | TricepDipDetector | elbow | depth, forward lean |

### BiLSTM Model F1 Scores (secondary, for comparison):
| Exercise | F1 | Dim | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| lunge | 0.786 | 330 | 0.765 | 0.809 | 0.25 |
| overhead_press | 0.756 | 330 | 0.758 | 0.754 | 0.40 |
| pullup | 0.751 | 109 | 0.727 | 0.777 | 0.30 |
| bicep_curl | 0.725 | 99 | 0.588 | 0.946 | 0.50 |
| bench_press | 0.718 | 99 | 0.657 | 0.791 | 0.45 |
| plank | 0.702 | 109 | 0.598 | 0.848 | 0.10 |
| pushup | 0.661 | 109 | 0.712 | 0.617 | 0.15 |
| squat | 0.647 | 99 | 0.545 | 0.798 | 0.15 |
| deadlift | 0.605 | 99 | 0.478 | 0.824 | 0.25 |
| tricep_dip | 0.597 | 330 | 0.486 | 0.775 | 0.50 |
| **Average** | **0.695** | | | | |

Note: Dedicated detectors are now the primary path. BiLSTM/ML are secondary options selectable in the guide page.

## Commands

```bash
# Run web app (Flask serves the built React SPA at /, legacy vanilla UI at /legacy)
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" app/server.py

# React frontend (Vite + React 19 + Tailwind 4 + GSAP + Framer Motion + Lenis)
cd app/web && npm install          # first time only
cd app/web && npm run dev          # dev server on :5173, proxies /api + /socket.io to Flask :5000
cd app/web && npm run build        # production build → app/static/dist/ (served by Flask)
cd app/web && npm run typecheck    # strict TS check, no emit

# UNIFIED PIPELINE — add data + retrain in one command
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/pipeline.py
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/pipeline.py --exercise squat lunge
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/pipeline.py --skip-extract  # skip landmark extraction
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/pipeline.py --dry-run       # show plan only

# Individual pipeline steps (if needed)
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/extract_landmarks_parallel.py --workers 28
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/split_videos.py --force
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/build_sequences.py --use-landmarks
"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe" scripts/train_bilstm.py --use-landmarks --exercise squat --epochs 500
```

## Coding Conventions

- All exercises referenced by lowercase underscore names: `squat`, `bench_press`, `bicep_curl`, etc.
- Landmark indices from `src/utils/constants.py` (e.g., LEFT_KNEE=25, RIGHT_HIP=24)
- World coordinates are hip-centered (MediaPipe default), in meters
- Image coordinates are normalized 0-1
- Joint angles computed at vertex: `calculate_angle(point_a, vertex, point_b)` returns degrees 0-180
- BGR color format for OpenCV rendering
- v1 models: `{exercise}_bilstm.pt` (old hand-crafted features, broken)
- v2 models: `{exercise}_bilstm_v2.pt` (new landmark features, CNN-BiLSTM)
- Feature CSVs have columns: exercise, label, source, video_id, then feature columns
- Landmark CSVs: 99 columns named lm_{landmark_name}_{x|y|z}
- Sequences stored as .npz with keys: X (n, seq_len, features), y (n,), video_ids, feature_names

## Important Rules

- ALWAYS quote file paths (project path has spaces)
- Use `"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"` to run Python
- When modifying `src/` core library: ensure backward compatibility with existing `app/server.py`
- The pipeline must gracefully fall back when v2 models don't exist
- When adding new features to LandmarkFeatureExtractor: keep backward compat with old FeatureExtractor
- When changing classification output format: update FeedbackEngine and overlay.py accordingly
- Test imports from project root: `sys.path.insert(0, str(PROJECT_ROOT))`

and always my concern is you have to use the most available compute so memory cpu shared gpu and dedicated gpu always alright

i am pretty sure i said it 1000 times to max out the use of cpu gpu and ram and yet every time youdon't

## MCP Servers (Available — USE when relevant)

The following MCP servers are installed globally and available in every conversation. **Use them proactively** when the task would benefit:

### Browser & Web Scraping
- **playwright** (`@playwright/mcp`) — Full browser automation via Microsoft Playwright. Navigate pages, click elements, fill forms, take screenshots, extract content from JS-rendered pages. **Use for**: testing the web app in a real browser, scraping exercise form data from fitness websites, automating any web interaction.
- **fetcher** (`fetcher-mcp`) — Playwright-based web page fetcher that returns clean markdown. **Use for**: grabbing content from any URL, reading documentation, fetching exercise tutorials or form guides from the web.

### YouTube & Video
- **youtube** (`@anaisbetts/mcp-youtube`) — YouTube video search, metadata extraction, and transcript downloading. **Use for**: finding new exercise form videos for training data, getting video metadata, searching for correct/incorrect form examples.
- **youtube-transcript** (`@sinco-lab/mcp-youtube-transcript`) — Direct YouTube transcript/caption downloading. **Use for**: analyzing video content without watching, extracting exercise instruction text from YouTube tutorials.
- **yt-dlp** (CLI tool, already installed via pip) — Download videos/audio from YouTube and 1000+ sites. Run via: `yt-dlp <url> -o "path/to/output"`. **Use for**: downloading new training videos for the dataset.

### Documentation & Knowledge
- **context7** (`@upstash/context7-mcp`) — Library documentation lookup. **Use for**: getting up-to-date docs for PyTorch, MediaPipe, Flask, Socket.IO, or any library used in the project.
- **sequential-thinking** (`@modelcontextprotocol/server-sequential-thinking`) — Structured reasoning tool for complex multi-step problems. **Use for**: debugging tricky issues, planning architecture changes, reasoning through data pipeline problems.

### When to use MCP servers
- Need to look up a URL or webpage content? → **fetcher** or **playwright**
- Need exercise form reference videos? → **youtube** search + **yt-dlp** download
- Need library documentation? → **context7**
- Need to test the web app UI? → **playwright**
- Complex reasoning needed? → **sequential-thinking**