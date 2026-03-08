"""
End-to-end integration tests for ExerVision.

Tests all components: pose estimation, feature extraction, classification,
feedback, visualization, and the full pipeline.

Usage:
    python scripts/test_pipeline.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper():
            global passed, failed
            try:
                func()
                passed += 1
                print(f"  [PASS] {name}")
            except Exception as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
        return wrapper
    return decorator


# ==================== Constants ====================

@test("Constants: EXERCISES has 10 items")
def test_constants_exercises():
    from src.utils.constants import EXERCISES
    assert len(EXERCISES) == 10, f"Expected 10, got {len(EXERCISES)}"

@test("Constants: LANDMARK_NAMES has 33 items")
def test_constants_landmarks():
    from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS
    assert len(LANDMARK_NAMES) == NUM_LANDMARKS == 33

@test("Constants: JOINT_ANGLES has 10 entries")
def test_constants_angles():
    from src.utils.constants import JOINT_ANGLES
    assert len(JOINT_ANGLES) == 10

@test("Constants: SKELETON_CONNECTIONS defined")
def test_constants_skeleton():
    from src.utils.constants import SKELETON_CONNECTIONS
    assert len(SKELETON_CONNECTIONS) >= 10


# ==================== Geometry ====================

@test("Geometry: calculate_angle returns degrees")
def test_geometry_angle():
    from src.utils.geometry import calculate_angle
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 0])
    c = np.array([0, 1, 0])
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1.0, f"Expected ~90, got {angle}"

@test("Geometry: distance works")
def test_geometry_distance():
    from src.utils.geometry import distance
    d = distance(np.array([0, 0, 0]), np.array([3, 4, 0]))
    assert abs(d - 5.0) < 0.01


# ==================== PoseResult ====================

@test("PoseResult: creation and methods")
def test_pose_result():
    from src.pose_estimation.base import PoseResult
    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9  # high visibility
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95, timestamp_ms=100)
    assert pose.get_landmark(0).shape == (3,)
    assert pose.get_world_landmark(0).shape == (3,)
    assert pose.is_visible(0, 0.5) == True


# ==================== Feature Extraction ====================

@test("FeatureExtractor: extract returns dict")
def test_feature_extractor():
    from src.pose_estimation.base import PoseResult
    from src.feature_extraction.features import FeatureExtractor

    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95)

    fe = FeatureExtractor(use_world=True)
    features = fe.extract(pose, "squat")
    assert isinstance(features, dict)
    assert "angle_left_knee" in features

@test("FeatureExtractor: extract_vector returns array")
def test_feature_vector():
    from src.pose_estimation.base import PoseResult
    from src.feature_extraction.features import FeatureExtractor

    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95)

    fe = FeatureExtractor(use_world=True)
    vec = fe.extract_vector(pose, "squat")
    assert isinstance(vec, np.ndarray)
    assert len(vec) > 0

@test("FeatureExtractor: all 10 exercises return features")
def test_all_exercises_features():
    from src.pose_estimation.base import PoseResult
    from src.feature_extraction.features import FeatureExtractor
    from src.utils.constants import EXERCISES

    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95)

    fe = FeatureExtractor(use_world=True)
    for exercise in EXERCISES:
        features = fe.extract(pose, exercise)
        assert len(features) > 0, f"No features for {exercise}"


# ==================== Classification ====================

@test("RuleBasedClassifier: returns ClassificationResult")
def test_rule_based():
    from src.pose_estimation.base import PoseResult
    from src.feature_extraction.features import FeatureExtractor
    from src.classification.rule_based import RuleBasedClassifier

    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95)

    fe = FeatureExtractor(use_world=True)
    clf = RuleBasedClassifier()
    features = fe.extract(pose, "squat")
    result = clf.classify(features, "squat")
    assert hasattr(result, "is_correct")
    assert hasattr(result, "confidence")
    assert hasattr(result, "joint_feedback")

@test("RuleBasedClassifier: supports all 10 exercises")
def test_rule_based_all():
    from src.classification.rule_based import RuleBasedClassifier
    clf = RuleBasedClassifier()
    assert len(clf.get_supported_exercises()) == 10

@test("MLClassifier: loads models if available")
def test_ml_classifier_load():
    from src.classification.ml_classifier import MLClassifier
    clf = MLClassifier()
    clf.load_all_models()
    # May or may not have models, just check it doesn't crash
    assert isinstance(clf.get_supported_exercises(), list)


# ==================== Feedback ====================

@test("FeedbackEngine: generates colors and text")
def test_feedback_engine():
    from src.classification.base import ClassificationResult
    from src.feedback.feedback_engine import FeedbackEngine

    result = ClassificationResult(
        exercise="squat",
        is_correct=True,
        confidence=0.9,
        joint_feedback={"left_knee": "correct", "right_knee": "incorrect"},
        details={"test": "test detail"},
    )
    engine = FeedbackEngine()
    colors = engine.get_joint_colors(result)
    text = engine.get_text_feedback(result)
    assert isinstance(colors, dict)
    assert isinstance(text, str)


# ==================== Temporal ====================

@test("TemporalSmoother: smoothes poses")
def test_temporal_smoother():
    from src.pose_estimation.base import PoseResult
    from src.utils.temporal import TemporalSmoother

    smoother = TemporalSmoother(window_size=3)
    for i in range(3):
        lm = np.random.rand(33, 4)
        lm[:, 3] = 0.9
        wlm = np.random.rand(33, 3)
        pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                          detection_confidence=0.9, timestamp_ms=i * 33)
        smoother.add(pose)

    assert smoother.is_ready
    smoothed = smoother.get_smoothed()
    assert smoothed is not None
    assert smoothed.landmarks.shape == (33, 4)

@test("RepCounter: counts reps")
def test_rep_counter():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("squat")
    assert counter.get_rep_count() == 0


# ==================== Augmentation ====================

@test("Augmentation: augment_pose returns list")
def test_augmentation():
    from src.pose_estimation.base import PoseResult
    from src.utils.augmentation import augment_pose

    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.9)

    augmented = augment_pose(pose, n_augmented=4)
    assert len(augmented) == 4
    for aug in augmented:
        assert aug.landmarks.shape == (33, 4)


# ==================== Visualization ====================

@test("Overlay: draw_skeleton works")
def test_overlay():
    import cv2
    from src.pose_estimation.base import PoseResult
    from src.visualization.overlay import draw_skeleton

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lm = np.random.rand(33, 4)
    lm[:, 3] = 0.9
    wlm = np.random.rand(33, 3)
    pose = PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.9)

    result = draw_skeleton(frame, pose)
    assert result.shape == (480, 640, 3)


# ==================== Evaluation ====================

@test("Evaluation metrics: evaluate_classifier works")
def test_eval_metrics():
    from src.evaluation.metrics import evaluate_classifier
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])
    metrics = evaluate_classifier(y_true, y_pred)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1


# ==================== Web App Imports ====================

@test("Flask app: imports without errors")
def test_flask_imports():
    import flask
    import flask_socketio
    # Just check imports work


# ==================== Data Files ====================

@test("Data: feature CSV exists")
def test_data_exists():
    features_dir = PROJECT_ROOT / "data" / "processed" / "features"
    csvs = list(features_dir.glob("*.csv"))
    assert len(csvs) > 0, "No feature CSVs found"

@test("Data: landmark CSVs exist")
def test_landmarks_exist():
    landmarks_dir = PROJECT_ROOT / "data" / "processed" / "landmarks"
    csvs = list(landmarks_dir.glob("*.csv"))
    assert len(csvs) > 0, f"No landmark CSVs found"


# ==================== Pipeline Integration ====================

def _make_synthetic_pose(knee_angle=90.0):
    """Create a PoseResult with controlled left knee angle.

    Places hip, knee, ankle in a plane so that calculate_angle(hip, knee, ankle)
    returns approximately `knee_angle` degrees.  All landmarks have high visibility.
    """
    from src.pose_estimation.base import PoseResult

    lm = np.zeros((33, 4))
    lm[:, 3] = 0.95  # visibility

    wlm = np.zeros((33, 3))

    # Indices: left_hip=23, left_knee=25, left_ankle=27
    # Place knee at origin, hip above, ankle at the desired angle
    rad = np.radians(knee_angle)
    # hip is above-left of knee
    hip_pos = np.array([0.0, -1.0, 0.0])
    # ankle is at the angle (measured at knee between hip-knee-ankle)
    ankle_pos = np.array([np.sin(np.pi - rad), -np.cos(np.pi - rad), 0.0])
    knee_pos = np.array([0.0, 0.0, 0.0])

    # Also set shoulder (index 11) above hip for feature extraction
    shoulder_pos = np.array([0.0, -2.0, 0.0])

    for idx, pos in [(23, hip_pos), (24, hip_pos), (25, knee_pos), (26, knee_pos),
                     (27, ankle_pos), (28, ankle_pos), (11, shoulder_pos), (12, shoulder_pos),
                     (13, shoulder_pos + [0.5, 0, 0]), (14, shoulder_pos + [-0.5, 0, 0]),
                     (15, shoulder_pos + [1.0, 0, 0]), (16, shoulder_pos + [-1.0, 0, 0])]:
        lm[idx, :3] = pos
        wlm[idx] = pos

    return PoseResult(landmarks=lm, world_landmarks=wlm,
                      detection_confidence=0.95, timestamp_ms=0)


@test("Pipeline: create with rule_based classifier")
def test_pipeline_create_rb():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    assert p.exercise == "squat"
    assert p.classifier_type == "rule_based"
    p.close()

@test("Pipeline: create with ml classifier")
def test_pipeline_create_ml():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "ml")
    assert p.classifier_type == "ml"
    p.close()

@test("Pipeline: create with bilstm classifier")
def test_pipeline_create_bilstm():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "bilstm")
    assert p.classifier_type == "bilstm"
    p.close()

@test("Pipeline: process_frame returns valid types")
def test_pipeline_process_frame():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result_frame, result = p.process_frame(frame, 0)
    assert isinstance(result_frame, np.ndarray)
    assert result_frame.shape[2] == 3  # BGR
    # result may be None if no pose detected on black frame -- that's fine
    p.close()

@test("Pipeline: classifier switching")
def test_pipeline_switch_classifier():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    p.set_classifier("ml")
    assert p.classifier_type == "ml"
    p.set_classifier("rule_based")
    assert p.classifier_type == "rule_based"
    p.close()

@test("Pipeline: exercise switching resets rep count")
def test_pipeline_switch_exercise():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    p.exercise = "lunge"
    assert p.exercise == "lunge"
    assert p.rep_count == 0
    p.close()

@test("Pipeline: session lifecycle returns summary")
def test_pipeline_session():
    import time
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    p.start_session()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(3):
        p.process_frame(frame, i * 33)
    time.sleep(0.05)  # ensure non-zero duration
    summary = p.end_session()
    assert isinstance(summary, dict)
    for key in ["total_reps", "good_reps", "avg_form_score", "duration_sec", "reps", "common_issues"]:
        assert key in summary, f"Missing key: {key}"
    assert summary["duration_sec"] >= 0
    p.close()

@test("Pipeline: reset clears state")
def test_pipeline_reset():
    from src.pipeline.realtime import ExerVisionPipeline
    p = ExerVisionPipeline("squat", "rule_based")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(3):
        p.process_frame(frame, i * 33)
    p.reset()
    assert p.rep_count == 0
    p.close()


# ==================== Rep Counter Logic ====================

@test("RepCounter: single squat rep detected")
def test_rep_counter_single():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("squat", min_angle_change=30.0)

    # Simulate: stand (170°) -> squat (80°) -> stand (170°) -> slight dip to trigger
    angles = list(range(170, 79, -5)) + list(range(80, 171, 5)) + list(range(170, 155, -5))
    for angle in angles:
        pose = _make_synthetic_pose(knee_angle=float(angle))
        counter.update(pose)

    assert counter.get_rep_count() == 1, f"Expected 1 rep, got {counter.get_rep_count()}"

@test("RepCounter: small ROM does not count")
def test_rep_counter_small_rom():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("squat", min_angle_change=30.0)

    # Only 20° change -- less than threshold
    angles = list(range(170, 150, -2)) + list(range(150, 171, 2))
    for angle in angles:
        pose = _make_synthetic_pose(knee_angle=float(angle))
        counter.update(pose)

    assert counter.get_rep_count() == 0, f"Expected 0 reps, got {counter.get_rep_count()}"

@test("RepCounter: three reps counted")
def test_rep_counter_three():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("squat", min_angle_change=30.0)

    for _ in range(3):
        for angle in list(range(170, 79, -5)) + list(range(80, 171, 5)):
            pose = _make_synthetic_pose(knee_angle=float(angle))
            counter.update(pose)
    # Small dip to trigger last rep boundary
    for angle in range(170, 155, -5):
        pose = _make_synthetic_pose(knee_angle=float(angle))
        counter.update(pose)

    assert counter.get_rep_count() == 3, f"Expected 3 reps, got {counter.get_rep_count()}"

@test("RepCounter: plank has no reps")
def test_rep_counter_plank():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("plank")
    for angle in list(range(170, 80, -5)) + list(range(80, 171, 5)):
        pose = _make_synthetic_pose(knee_angle=float(angle))
        counter.update(pose)
    assert counter.get_rep_count() == 0

@test("RepCounter: reset clears count")
def test_rep_counter_reset():
    from src.feature_extraction.rep_counter import RepCounter
    counter = RepCounter("squat", min_angle_change=30.0)
    for angle in list(range(170, 79, -5)) + list(range(80, 171, 5)) + list(range(170, 155, -5)):
        pose = _make_synthetic_pose(knee_angle=float(angle))
        counter.update(pose)
    assert counter.get_rep_count() >= 1
    counter.reset()
    assert counter.get_rep_count() == 0


# ==================== Flask API ====================

@test("Flask API: /api/exercises returns 10 exercises")
def test_flask_exercises():
    from app.server import app
    client = app.test_client()
    resp = client.get("/api/exercises")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "exercises" in data
    assert len(data["exercises"]) == 10

@test("Flask API: exercise list has all 10 IDs")
def test_flask_exercise_ids():
    from app.server import app
    from src.utils.constants import EXERCISES
    client = app.test_client()
    data = client.get("/api/exercises").get_json()
    ids = {e["id"] for e in data["exercises"]}
    for ex in EXERCISES:
        assert ex in ids, f"Missing exercise: {ex}"

@test("Flask API: exercise guide returns valid data")
def test_flask_guide():
    from app.server import app
    client = app.test_client()
    resp = client.get("/api/exercise_guide/squat")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "display_name" in data
    assert "muscles" in data
    assert "instructions" in data
    assert "common_mistakes" in data

@test("Flask API: guide works for all 10 exercises")
def test_flask_guide_all():
    from app.server import app
    from src.utils.constants import EXERCISES
    client = app.test_client()
    for ex in EXERCISES:
        resp = client.get(f"/api/exercise_guide/{ex}")
        assert resp.status_code == 200, f"Guide failed for {ex}: status {resp.status_code}"

@test("Flask API: guide structure has lists")
def test_flask_guide_structure():
    from app.server import app
    client = app.test_client()
    data = client.get("/api/exercise_guide/squat").get_json()
    assert isinstance(data["muscles"], list)
    assert isinstance(data["instructions"], list)
    assert isinstance(data["common_mistakes"], list)

@test("Flask API: invalid exercise returns 404")
def test_flask_invalid_exercise():
    from app.server import app
    client = app.test_client()
    resp = client.get("/api/exercise_guide/nonexistent_exercise")
    assert resp.status_code == 404


# ==================== Model & Data Integrity ====================

@test("Models: all 10 RF models exist")
def test_rf_models_exist():
    from src.utils.constants import EXERCISES
    import pickle
    for ex in EXERCISES:
        path = PROJECT_ROOT / "models" / "trained" / f"{ex}_classifier.pkl"
        assert path.exists(), f"Missing RF model: {path.name}"

@test("Models: RF models are loadable with predict")
def test_rf_models_loadable():
    import pickle
    from src.utils.constants import EXERCISES
    for ex in EXERCISES:
        path = PROJECT_ROOT / "models" / "trained" / f"{ex}_classifier.pkl"
        with open(path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict"), f"{ex} model has no predict method"

@test("Models: all 10 BiLSTM models exist")
def test_bilstm_models_exist():
    from src.utils.constants import EXERCISES
    for ex in EXERCISES:
        path = PROJECT_ROOT / "models" / "trained" / f"{ex}_bilstm.pt"
        assert path.exists(), f"Missing BiLSTM model: {path.name}"

@test("Models: BiLSTM checkpoints have required keys")
def test_bilstm_checkpoints():
    import torch
    from src.utils.constants import EXERCISES
    required_keys = {"model_state_dict", "input_dim", "hidden_dim", "feature_names"}
    for ex in EXERCISES:
        path = PROJECT_ROOT / "models" / "trained" / f"{ex}_bilstm.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for key in required_keys:
            assert key in ckpt, f"{ex} checkpoint missing key: {key}"

@test("Data: features CSV has all 10 exercises")
def test_features_all_exercises():
    import pandas as pd
    from src.utils.constants import EXERCISES
    path = PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv"
    assert path.exists(), "all_features.csv not found"
    df = pd.read_csv(path, usecols=["exercise"])
    exercises_in_csv = set(df["exercise"].unique())
    for ex in EXERCISES:
        assert ex in exercises_in_csv, f"Missing exercise in CSV: {ex}"

@test("Data: sequence files exist for all 10 exercises")
def test_sequences_exist():
    from src.utils.constants import EXERCISES
    for ex in EXERCISES:
        path = PROJECT_ROOT / "data" / "processed" / "sequences" / f"{ex}_sequences.npz"
        assert path.exists(), f"Missing sequence file: {path.name}"
        data = np.load(path, allow_pickle=True)
        for key in ["X", "y", "video_ids"]:
            assert key in data, f"{ex} sequences missing key: {key}"


# ==================== Run All ====================

def main():
    global passed, failed

    print("=" * 60)
    print("ExerVision Integration Tests")
    print("=" * 60)

    tests = [
        # Constants
        test_constants_exercises,
        test_constants_landmarks,
        test_constants_angles,
        test_constants_skeleton,
        # Geometry
        test_geometry_angle,
        test_geometry_distance,
        # PoseResult
        test_pose_result,
        # Feature extraction
        test_feature_extractor,
        test_feature_vector,
        test_all_exercises_features,
        # Classification
        test_rule_based,
        test_rule_based_all,
        test_ml_classifier_load,
        # Feedback
        test_feedback_engine,
        # Temporal
        test_temporal_smoother,
        test_rep_counter,
        # Augmentation
        test_augmentation,
        # Visualization
        test_overlay,
        # Evaluation
        test_eval_metrics,
        # Web imports
        test_flask_imports,
        # Data
        test_data_exists,
        test_landmarks_exist,
        # Pipeline integration
        test_pipeline_create_rb,
        test_pipeline_create_ml,
        test_pipeline_create_bilstm,
        test_pipeline_process_frame,
        test_pipeline_switch_classifier,
        test_pipeline_switch_exercise,
        test_pipeline_session,
        test_pipeline_reset,
        # Rep counter logic
        test_rep_counter_single,
        test_rep_counter_small_rom,
        test_rep_counter_three,
        test_rep_counter_plank,
        test_rep_counter_reset,
        # Flask API
        test_flask_exercises,
        test_flask_exercise_ids,
        test_flask_guide,
        test_flask_guide_all,
        test_flask_guide_structure,
        test_flask_invalid_exercise,
        # Model & data integrity
        test_rf_models_exist,
        test_rf_models_loadable,
        test_bilstm_models_exist,
        test_bilstm_checkpoints,
        test_features_all_exercises,
        test_sequences_exist,
    ]

    for test_fn in tests:
        test_fn()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
