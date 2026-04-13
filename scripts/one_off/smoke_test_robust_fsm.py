"""
Smoke test for RobustExerciseDetector subclasses.

Feeds a synthetic 30-frame primary-angle sequence through each specified
detector and verifies exactly 1 rep commits. Used to validate the robust
rep FSM before running a live session.

Usage:
    python scripts/one_off/smoke_test_robust_fsm.py squat deadlift bicep_curl
    python scripts/one_off/smoke_test_robust_fsm.py all
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.classification.base_detector import (  # noqa: E402
    RobustExerciseDetector,
    SessionState,
    REP_DIRECTION_DECREASING,
    REP_DIRECTION_INCREASING,
)
from src.classification.posture_classifier import PostureLabel  # noqa: E402
from src.pose_estimation.base import PoseResult  # noqa: E402
from src.utils.constants import (  # noqa: E402
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_EAR, RIGHT_EAR,
    NOSE,
)


def _blank_pose() -> tuple[np.ndarray, np.ndarray]:
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 1.0
    wlm = np.zeros((33, 3), dtype=np.float32)
    return lm, wlm


def _set_elbow(wlm: np.ndarray, elbow_deg: float, set_shoulder: bool = True) -> None:
    """Place L/R shoulder-elbow-wrist so each elbow joint = elbow_deg exactly.

    Math: with upper arm pointing straight down from shoulder to elbow
    (length L), the forearm (E→W) is rotated from the upper arm direction
    by elbow_deg in the xy plane. For angle elbow_deg at the vertex E:
        (W - E).x = L * sin(elbow_deg)
        (W - E).y = -L * cos(elbow_deg)

    At 180°: (0, L, 0) — wrist below elbow, arm straight. ✓
    At  90°: (L, 0, 0) — forearm horizontal. ✓
    At   0°: (0, -L, 0) — forearm folded back to shoulder. ✓

    If `set_shoulder=False`, uses the shoulder position already set in
    wlm (for composing with _set_hip which owns shoulder placement).
    """
    L = 0.30  # arm segment length in meters
    r = np.radians(elbow_deg)
    forearm_dx = L * np.sin(r)
    forearm_dy = -L * np.cos(r)
    for sh, el, wr in [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ]:
        if set_shoulder:
            wlm[sh] = [0.0, -0.4, 0.0]
        shoulder = wlm[sh]
        elbow = np.array([shoulder[0], shoulder[1] + L, shoulder[2]])
        wlm[el] = elbow
        wlm[wr] = np.array([
            elbow[0] + forearm_dx,
            elbow[1] + forearm_dy,
            elbow[2],
        ])


def _set_knee(wlm: np.ndarray, knee_deg: float) -> None:
    """Place L/R hip-knee-ankle so each knee joint = knee_deg exactly.

    Approach: fix hip at origin and knee 40cm below. The shin vector
    (knee→ankle) rotates from straight-down (knee_deg=180°) through
    horizontal (knee_deg=90°) toward straight-up (knee_deg=0°), such
    that the angle between thigh and shin at the knee = knee_deg.

    Math: thigh = (0, -0.4, 0). For angle knee_deg between thigh and
    shin:
        shin.y = -shin_len * cos(knee_deg)       (vertical component)
        shin.x =  shin_len * sin(knee_deg)       (horizontal component)
    At 180°: shin=(0, +shin_len, 0) ankle directly below. ✓
    At  90°: shin=(+shin_len, 0, 0) ankle horizontally back. ✓
    At  80°: shin=(0.985*shin_len, -0.174*shin_len, 0) ankle slightly
             above knee (physically = heel rising in a very deep squat).
    """
    r = np.radians(knee_deg)
    shin_len = 0.40
    dy = -shin_len * np.cos(r)
    dx = shin_len * np.sin(r)
    for hp, kn, an in [
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    ]:
        wlm[hp] = [0.0, 0.0, 0.0]
        wlm[kn] = [0.0, 0.4, 0.0]
        wlm[an] = [dx, 0.4 + dy, 0.0]


def _set_hip(wlm: np.ndarray, hip_deg: float) -> None:
    """Place L/R shoulder-hip-knee so each hip joint = hip_deg exactly.

    Math: at the hip vertex, angle between (hip→knee) and (hip→shoulder)
    must equal hip_deg. Knee is fixed 40cm below hip (thigh vector
    (0, 0.4, 0) in y-down coords). Shoulder = thigh vector rotated by
    hip_deg in the xy plane:
        back.x = -0.4 * sin(hip_deg)
        back.y =  0.4 * cos(hip_deg)

    At 180°: back = (0, -0.4, 0), shoulder directly above hip. ✓
    At  90°: back = (-0.4, 0, 0), torso horizontal forward. ✓
    At  85°: back = (-0.398, 0.035, 0), torso slightly past horizontal.
    """
    r = np.radians(hip_deg)
    for sh, hp, kn in [
        (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ]:
        wlm[hp] = [0.0, 0.0, 0.0]
        wlm[kn] = [0.0, 0.4, 0.0]
        wlm[sh] = [
            -0.4 * np.sin(r),
            0.4 * np.cos(r),
            0.0,
        ]
    for an in (LEFT_ANKLE, RIGHT_ANKLE):
        wlm[an] = [0.0, 0.6, 0.0]


def _set_head_and_visibility(lm: np.ndarray, wlm: np.ndarray) -> None:
    """Face/ears default positions (for back-ratio + posture features)."""
    wlm[NOSE] = [0.0, -0.35, 0.0]
    wlm[LEFT_EAR] = [-0.05, -0.35, 0.0]
    wlm[RIGHT_EAR] = [0.05, -0.35, 0.0]
    for i in range(33):
        lm[i, 0] = wlm[i, 0]
        lm[i, 1] = wlm[i, 1]
        lm[i, 2] = wlm[i, 2]


def _build_pose(lm: np.ndarray, wlm: np.ndarray) -> PoseResult:
    return PoseResult(
        landmarks=lm,
        world_landmarks=wlm,
        detection_confidence=0.95,
        timestamp_ms=0,
    )


def pose_elbow_only(elbow_deg: float) -> PoseResult:
    """Pose for exercises where primary=elbow (pushup, bench, ohp, pullup,
    bicep curl, tricep dip). Legs held in extended standing position so
    posture gates pass."""
    lm, wlm = _blank_pose()
    _set_knee(wlm, 175.0)       # standing legs
    _set_elbow(wlm, elbow_deg)   # variable elbow
    _set_head_and_visibility(lm, wlm)
    return _build_pose(lm, wlm)


def pose_knee_only(knee_deg: float) -> PoseResult:
    """Pose for exercises where primary=knee (squat, lunge)."""
    lm, wlm = _blank_pose()
    _set_knee(wlm, knee_deg)
    _set_elbow(wlm, 170.0)       # arms extended at sides
    _set_head_and_visibility(lm, wlm)
    return _build_pose(lm, wlm)


def pose_hip_only(hip_deg: float) -> PoseResult:
    """Pose for exercises where primary=hip (deadlift)."""
    lm, wlm = _blank_pose()
    _set_hip(wlm, hip_deg)
    # Don't overwrite shoulder — _set_hip already placed it for the
    # target hip angle. Elbow/wrist go relative to that shoulder.
    _set_elbow(wlm, 170.0, set_shoulder=False)
    _set_head_and_visibility(lm, wlm)
    return _build_pose(lm, wlm)


def build_primary_sequence(start: float, bottom: float, n_frames: int = 30) -> list[float]:
    """Build a symmetric up→down→up primary-angle sequence.

    5 frames at top → 10 descending → 5 at bottom → 10 ascending.
    """
    top_hold = [start] * 5
    descending = list(np.linspace(start, bottom, 10)[1:])  # 9 frames
    bottom_hold = [bottom] * 5
    ascending = list(np.linspace(bottom, start, 10)[1:])  # 9 frames
    return top_hold + descending + bottom_hold + ascending  # 28 frames


def run_detector_test(detector_cls, exercise_name: str, pose_builder,
                      primary_start: float, primary_bottom: float,
                      bypass_posture: bool = True) -> dict:
    """Instantiate a detector, feed synthetic poses, return a test report.

    bypass_posture=True monkey-patches the session FSM to ACTIVE so we
    isolate the rep FSM behavior. Without this, the posture classifier
    would need 5+ frames to bootstrap and the synthetic sequence is too
    short to cleanly test rep FSM in isolation.
    """
    detector = detector_cls()
    detector.reset()

    if bypass_posture:
        detector._session_state = SessionState.ACTIVE
        detector._form_validated = True

    primary_seq = build_primary_sequence(primary_start, primary_bottom, n_frames=28)

    reps_before = detector.rep_count
    commits: list[int] = []
    states: list[str] = []

    for i, primary in enumerate(primary_seq):
        pose = pose_builder(primary)
        result = detector.classify(pose, timestamp=i * 0.1)
        states.append(detector._state)
        if detector.rep_count > reps_before + len(commits):
            commits.append(i)

    reps_after = detector.rep_count
    expected = 1

    return {
        "exercise": exercise_name,
        "reps_before": reps_before,
        "reps_after": reps_after,
        "expected": expected,
        "passed": reps_after - reps_before == expected,
        "commit_frames": commits,
        "final_state": detector._state,
        "final_session_state": detector._session_state.value,
        "primary_start": primary_start,
        "primary_bottom": primary_bottom,
    }


def load_detector(name: str):
    """Return the detector class for a given exercise name, if it has been
    migrated to RobustExerciseDetector. Otherwise return None."""
    mapping = {
        "squat": ("src.classification.squat_detector", "SquatDetector"),
        "deadlift": ("src.classification.deadlift_detector", "DeadliftDetector"),
        "bicep_curl": ("src.classification.bicep_curl_detector", "BicepCurlDetector"),
        "lunge": ("src.classification.lunge_detector", "LungeDetector"),
        "overhead_press": ("src.classification.overhead_press_detector", "OverheadPressDetector"),
        "bench_press": ("src.classification.bench_press_detector", "BenchPressDetector"),
        "pullup": ("src.classification.pullup_detector", "PullUpDetector"),
        "tricep_dip": ("src.classification.tricep_dip_detector", "TricepDipDetector"),
        "plank": ("src.classification.plank_detector", "PlankDetector"),
    }
    if name not in mapping:
        return None
    module_path, cls_name = mapping[name]
    try:
        module = __import__(module_path, fromlist=[cls_name])
        return getattr(module, cls_name)
    except Exception as e:
        print(f"  load error: {e}")
        return None


# Per-exercise synthetic test config:
#   (pose_builder_name, primary_start, primary_bottom, bypass_posture)
# pose_builder chooses which joint gets the variable angle; the rest are
# held in a standing/ready position so posture gates pass.
EXERCISE_CONFIG = {
    "squat":          ("knee",  175.0,  80.0, True),
    "deadlift":       ("hip",   165.0,  80.0, True),
    "bicep_curl":     ("elbow", 170.0,  40.0, True),
    "lunge":          ("knee",  175.0,  85.0, True),
    "overhead_press": ("elbow",  85.0, 170.0, True),  # INCREASING
    "bench_press":    ("elbow", 170.0,  85.0, True),
    "pullup":         ("elbow", 175.0,  75.0, True),
    "tricep_dip":     ("elbow", 165.0,  85.0, True),
}

POSE_BUILDERS = {
    "elbow": pose_elbow_only,
    "knee": pose_knee_only,
    "hip": pose_hip_only,
}


def main():
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        exercises = list(EXERCISE_CONFIG.keys())
    else:
        exercises = args

    print(f"Running robust FSM smoke test on: {', '.join(exercises)}")
    print("=" * 60)

    results = []
    for ex in exercises:
        cls = load_detector(ex)
        if cls is None:
            print(f"[SKIP] {ex}: detector not available")
            continue
        if not issubclass(cls, RobustExerciseDetector):
            print(f"[SKIP] {ex}: not yet migrated to RobustExerciseDetector")
            continue

        cfg = EXERCISE_CONFIG.get(ex)
        if cfg is None:
            print(f"[SKIP] {ex}: no synthetic config")
            continue

        builder_key, primary_start, primary_bottom, bypass = cfg
        pose_builder = POSE_BUILDERS[builder_key]
        result = run_detector_test(
            cls, ex, pose_builder,
            primary_start, primary_bottom, bypass_posture=bypass,
        )
        results.append(result)
        tag = "PASS" if result["passed"] else "FAIL"
        print(f"[{tag}] {ex}: reps {result['reps_before']} -> {result['reps_after']} "
              f"(expected +{result['expected']})")
        if not result["passed"]:
            print(f"       primary {result['primary_start']} -> {result['primary_bottom']}")
            print(f"       final state={result['final_state']} session={result['final_session_state']}")
            print(f"       commits at frames: {result['commit_frames']}")

    print("=" * 60)
    n_pass = sum(1 for r in results if r["passed"])
    n_total = len(results)
    print(f"Result: {n_pass}/{n_total} passed")

    if n_total == 0:
        print("No detectors tested (maybe none migrated yet). Exiting 0.")
        sys.exit(0)
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
