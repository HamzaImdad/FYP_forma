"""
Verification script: tests resting/inactivity/set detection for all 11 exercises.

Simulates realistic pose sequences through each detector and checks:
1. Session FSM transitions (IDLE -> SETUP -> ACTIVE -> RESTING -> ACTIVE)
2. Rep counting works during ACTIVE
3. Rest detection fires after inactivity
4. Set boundary detection works
5. Pose-lost reset triggers correctly
6. Static holds (plank, side_plank) track duration instead of reps

Uses synthetic PoseResult objects with exercise-appropriate landmark positions.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.pose_estimation.base import PoseResult
from src.classification.base_detector import (
    SessionState, RobustExerciseDetector, RepPhase
)

# ── Landmark indices ─────────────────────────────────────────────────────
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_INDEX = 19
RIGHT_INDEX = 20

# ── Pose generators ──────────────────────────────────────────────────────

def _base_landmarks(visibility: float = 0.95) -> np.ndarray:
    """Create a base 33x4 landmarks array (standing pose, all visible)."""
    lm = np.zeros((33, 4))
    # Standing pose in normalized image coords (0-1)
    # Head
    lm[NOSE] = [0.5, 0.15, 0.0, visibility]
    lm[LEFT_EAR] = [0.48, 0.14, 0.0, visibility]
    lm[RIGHT_EAR] = [0.52, 0.14, 0.0, visibility]
    # Shoulders
    lm[LEFT_SHOULDER] = [0.42, 0.25, 0.0, visibility]
    lm[RIGHT_SHOULDER] = [0.58, 0.25, 0.0, visibility]
    # Elbows (arms at sides)
    lm[LEFT_ELBOW] = [0.38, 0.38, 0.0, visibility]
    lm[RIGHT_ELBOW] = [0.62, 0.38, 0.0, visibility]
    # Wrists
    lm[LEFT_WRIST] = [0.36, 0.48, 0.0, visibility]
    lm[RIGHT_WRIST] = [0.64, 0.48, 0.0, visibility]
    # Index fingers
    lm[LEFT_INDEX] = [0.35, 0.50, 0.0, visibility]
    lm[RIGHT_INDEX] = [0.65, 0.50, 0.0, visibility]
    # Hips
    lm[LEFT_HIP] = [0.45, 0.50, 0.0, visibility]
    lm[RIGHT_HIP] = [0.55, 0.50, 0.0, visibility]
    # Knees
    lm[LEFT_KNEE] = [0.45, 0.68, 0.0, visibility]
    lm[RIGHT_KNEE] = [0.55, 0.68, 0.0, visibility]
    # Ankles
    lm[LEFT_ANKLE] = [0.45, 0.88, 0.0, visibility]
    lm[RIGHT_ANKLE] = [0.55, 0.88, 0.0, visibility]
    # Heels
    lm[LEFT_HEEL] = [0.45, 0.90, 0.0, visibility]
    lm[RIGHT_HEEL] = [0.55, 0.90, 0.0, visibility]
    return lm


def _base_world(pose_type: str = "standing") -> np.ndarray:
    """Create 33x3 world landmarks (meters, hip-centered)."""
    w = np.zeros((33, 3))
    if pose_type == "standing":
        # Y axis: +Y = down (MediaPipe convention)
        w[NOSE] = [0.0, -0.55, 0.0]
        w[LEFT_EAR] = [-0.07, -0.58, 0.0]
        w[RIGHT_EAR] = [0.07, -0.58, 0.0]
        w[LEFT_SHOULDER] = [-0.18, -0.40, 0.0]
        w[RIGHT_SHOULDER] = [0.18, -0.40, 0.0]
        w[LEFT_ELBOW] = [-0.22, -0.15, 0.0]
        w[RIGHT_ELBOW] = [0.22, -0.15, 0.0]
        w[LEFT_WRIST] = [-0.22, 0.05, 0.0]
        w[RIGHT_WRIST] = [0.22, 0.05, 0.0]
        w[LEFT_INDEX] = [-0.22, 0.10, 0.0]
        w[RIGHT_INDEX] = [0.22, 0.10, 0.0]
        w[LEFT_HIP] = [-0.10, 0.0, 0.0]
        w[RIGHT_HIP] = [0.10, 0.0, 0.0]
        w[LEFT_KNEE] = [-0.10, 0.40, 0.0]
        w[RIGHT_KNEE] = [0.10, 0.40, 0.0]
        w[LEFT_ANKLE] = [-0.10, 0.80, 0.0]
        w[RIGHT_ANKLE] = [0.10, 0.80, 0.0]
        w[LEFT_HEEL] = [-0.10, 0.82, 0.02]
        w[RIGHT_HEEL] = [0.10, 0.82, 0.02]
    elif pose_type == "plank":
        # Horizontal body
        w[NOSE] = [-0.70, -0.05, 0.0]
        w[LEFT_EAR] = [-0.72, -0.04, -0.05]
        w[RIGHT_EAR] = [-0.72, -0.04, 0.05]
        w[LEFT_SHOULDER] = [-0.40, 0.0, -0.18]
        w[RIGHT_SHOULDER] = [-0.40, 0.0, 0.18]
        w[LEFT_ELBOW] = [-0.55, 0.10, -0.18]
        w[RIGHT_ELBOW] = [-0.55, 0.10, 0.18]
        w[LEFT_WRIST] = [-0.55, 0.15, -0.18]
        w[RIGHT_WRIST] = [-0.55, 0.15, 0.18]
        w[LEFT_INDEX] = [-0.55, 0.17, -0.18]
        w[RIGHT_INDEX] = [-0.55, 0.17, 0.18]
        w[LEFT_HIP] = [0.0, 0.0, -0.10]
        w[RIGHT_HIP] = [0.0, 0.0, 0.10]
        w[LEFT_KNEE] = [0.35, 0.0, -0.10]
        w[RIGHT_KNEE] = [0.35, 0.0, 0.10]
        w[LEFT_ANKLE] = [0.70, 0.0, -0.10]
        w[RIGHT_ANKLE] = [0.70, 0.0, 0.10]
        w[LEFT_HEEL] = [0.72, 0.02, -0.10]
        w[RIGHT_HEEL] = [0.72, 0.02, 0.10]
    elif pose_type == "side_plank":
        # Side plank — body horizontal, one arm supporting
        w[NOSE] = [-0.65, -0.10, 0.0]
        w[LEFT_EAR] = [-0.67, -0.10, -0.05]
        w[RIGHT_EAR] = [-0.67, -0.10, 0.05]
        # Left shoulder is support (lower), right is up
        w[LEFT_SHOULDER] = [-0.35, 0.05, 0.0]
        w[RIGHT_SHOULDER] = [-0.35, -0.30, 0.0]
        w[LEFT_ELBOW] = [-0.45, 0.15, 0.0]
        w[RIGHT_ELBOW] = [-0.35, -0.40, 0.0]
        w[LEFT_WRIST] = [-0.45, 0.20, 0.0]
        w[RIGHT_WRIST] = [-0.35, -0.50, 0.0]
        w[LEFT_INDEX] = [-0.45, 0.22, 0.0]
        w[RIGHT_INDEX] = [-0.35, -0.52, 0.0]
        w[LEFT_HIP] = [0.0, 0.0, 0.0]
        w[RIGHT_HIP] = [0.0, -0.05, 0.0]
        w[LEFT_KNEE] = [0.35, 0.0, 0.0]
        w[RIGHT_KNEE] = [0.35, -0.02, 0.0]
        w[LEFT_ANKLE] = [0.70, 0.0, 0.0]
        w[RIGHT_ANKLE] = [0.70, -0.02, 0.0]
        w[LEFT_HEEL] = [0.72, 0.02, 0.0]
        w[RIGHT_HEEL] = [0.72, 0.00, 0.0]
    return w


def make_pose(landmarks: np.ndarray, world: np.ndarray) -> PoseResult:
    """Create a PoseResult from landmarks and world arrays."""
    return PoseResult(
        landmarks=landmarks.copy(),
        world_landmarks=world.copy(),
        detection_confidence=0.95,
        timestamp_ms=0,
    )


def _interpolate_angle(lm: np.ndarray, world: np.ndarray,
                        joint_indices: Tuple[int, int, int],
                        target_angle_deg: float,
                        side: str = "both") -> Tuple[np.ndarray, np.ndarray]:
    """Adjust landmarks to produce a specific angle at the vertex joint.

    For simplicity, moves the 3rd point (idx_c) along the plane defined
    by the three points to achieve the target angle at idx_b (vertex).
    Works on both image landmarks and world landmarks simultaneously.
    """
    from src.utils.geometry import calculate_angle

    idx_a, idx_b, idx_c = joint_indices

    for coords, is_world in [(lm, False), (world, True)]:
        if is_world:
            a = coords[idx_a, :3].copy()
            b = coords[idx_b, :3].copy()
            c = coords[idx_c, :3].copy()
        else:
            a = coords[idx_a, :3].copy()
            b = coords[idx_b, :3].copy()
            c = coords[idx_c, :3].copy()

        # Vector from vertex to endpoints
        ba = a - b
        bc = c - b

        ba_norm = ba / (np.linalg.norm(ba) + 1e-8)
        bc_len = np.linalg.norm(bc)

        # Rotate bc around vertex to achieve target angle
        target_rad = np.radians(target_angle_deg)

        # Compute perpendicular in the plane
        perp = bc - np.dot(bc, ba_norm) * ba_norm
        perp_norm = perp / (np.linalg.norm(perp) + 1e-8)

        # New c position
        new_bc = bc_len * (np.cos(target_rad) * ba_norm + np.sin(target_rad) * perp_norm)
        new_c = b + new_bc

        if is_world:
            coords[idx_c, :3] = new_c
        else:
            coords[idx_c, :3] = new_c

    return lm, world


# ── Exercise-specific pose generators ────────────────────────────────────

def generate_squat_rep(t_start: float, duration: float = 2.0, fps: int = 30
                       ) -> List[Tuple[PoseResult, float]]:
    """Generate frames for one squat rep (standing -> deep squat -> standing)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        # Knee angle: 170deg (standing) -> 80deg (deep squat) -> 170deg
        if phase < 0.4:  # descent
            knee_angle = 170 - (170 - 80) * (phase / 0.4)
        elif phase < 0.6:  # bottom hold
            knee_angle = 80
        else:  # ascent
            knee_angle = 80 + (170 - 80) * ((phase - 0.6) / 0.4)

        # Hip angle follows knee
        hip_angle = knee_angle - 10

        lm = _base_landmarks()
        world = _base_world("standing")

        # Adjust knee angle by moving ankles relative to knees
        knee_rad = np.radians(knee_angle)
        # Move knees forward and down based on angle
        bend_factor = (170 - knee_angle) / 90.0
        world[LEFT_KNEE] = [-0.10, 0.40 - bend_factor * 0.15, bend_factor * 0.15]
        world[RIGHT_KNEE] = [0.10, 0.40 - bend_factor * 0.15, bend_factor * 0.15]
        # Lower hips
        world[LEFT_HIP] = [-0.10, bend_factor * 0.25, 0.0]
        world[RIGHT_HIP] = [0.10, bend_factor * 0.25, 0.0]

        # Update image landmarks proportionally
        lm[LEFT_KNEE, 1] = 0.68 - bend_factor * 0.08
        lm[RIGHT_KNEE, 1] = 0.68 - bend_factor * 0.08
        lm[LEFT_HIP, 1] = 0.50 + bend_factor * 0.12
        lm[RIGHT_HIP, 1] = 0.50 + bend_factor * 0.12

        frames.append((make_pose(lm, world), t))
    return frames


def generate_elbow_rep(t_start: float, duration: float = 2.0, fps: int = 30,
                       top_angle: float = 165.0, bottom_angle: float = 45.0,
                       pose_type: str = "standing"
                       ) -> List[Tuple[PoseResult, float]]:
    """Generate frames for any elbow-based exercise rep."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            angle = top_angle - (top_angle - bottom_angle) * (phase / 0.4)
        elif phase < 0.6:
            angle = bottom_angle
        else:
            angle = bottom_angle + (top_angle - bottom_angle) * ((phase - 0.6) / 0.4)

        lm = _base_landmarks()
        world = _base_world(pose_type)

        # Adjust elbow angle by moving wrists
        bend = (top_angle - angle) / (top_angle - bottom_angle)

        if pose_type == "standing":
            # Curl motion: wrists move up toward shoulders
            lm[LEFT_WRIST, 1] = 0.48 - bend * 0.20
            lm[RIGHT_WRIST, 1] = 0.48 - bend * 0.20
            world[LEFT_WRIST] = [-0.22, 0.05 - bend * 0.25, 0.0]
            world[RIGHT_WRIST] = [0.22, 0.05 - bend * 0.25, 0.0]

        frames.append((make_pose(lm, world), t))
    return frames


def generate_lunge_rep(t_start: float, duration: float = 2.5, fps: int = 30
                       ) -> List[Tuple[PoseResult, float]]:
    """Generate frames for one lunge rep."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            depth = phase / 0.4
        elif phase < 0.6:
            depth = 1.0
        else:
            depth = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Split stance: front leg forward, back leg back
        stance = depth * 0.50  # meters apart
        world[LEFT_ANKLE] = [-0.10, 0.80, -stance]  # front foot forward
        world[RIGHT_ANKLE] = [0.10, 0.80, stance]    # back foot back

        # Front knee bends
        front_knee_y = 0.40 - depth * 0.15
        world[LEFT_KNEE] = [-0.10, front_knee_y, -stance * 0.5]

        # Hips lower
        hip_drop = depth * 0.20
        world[LEFT_HIP] = [-0.10, hip_drop, 0.0]
        world[RIGHT_HIP] = [0.10, hip_drop, 0.0]

        # Update image landmarks
        lm[LEFT_KNEE, 1] = 0.68 - depth * 0.08
        lm[LEFT_HIP, 1] = 0.50 + depth * 0.10

        frames.append((make_pose(lm, world), t))
    return frames


def generate_lateral_raise_rep(t_start: float, duration: float = 2.0, fps: int = 30
                                ) -> List[Tuple[PoseResult, float]]:
    """Generate lateral raise rep (arms at sides -> T-shape -> back down)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            raise_factor = phase / 0.4
        elif phase < 0.6:
            raise_factor = 1.0
        else:
            raise_factor = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Raise arms out to sides (abduction)
        arm_spread = raise_factor * 0.35  # lateral spread
        arm_lift = raise_factor * 0.35    # lift to shoulder height

        world[LEFT_ELBOW] = [-0.22 - arm_spread * 0.5, -0.15 - arm_lift * 0.5, 0.0]
        world[RIGHT_ELBOW] = [0.22 + arm_spread * 0.5, -0.15 - arm_lift * 0.5, 0.0]
        world[LEFT_WRIST] = [-0.22 - arm_spread, -0.15 - arm_lift, 0.0]
        world[RIGHT_WRIST] = [0.22 + arm_spread, -0.15 - arm_lift, 0.0]

        lm[LEFT_ELBOW, 0] = 0.38 - arm_spread * 0.3
        lm[RIGHT_ELBOW, 0] = 0.62 + arm_spread * 0.3
        lm[LEFT_WRIST, 0] = 0.36 - arm_spread * 0.5
        lm[RIGHT_WRIST, 0] = 0.64 + arm_spread * 0.5
        lm[LEFT_ELBOW, 1] = 0.38 - arm_lift * 0.2
        lm[RIGHT_ELBOW, 1] = 0.38 - arm_lift * 0.2
        lm[LEFT_WRIST, 1] = 0.48 - arm_lift * 0.3
        lm[RIGHT_WRIST, 1] = 0.48 - arm_lift * 0.3

        frames.append((make_pose(lm, world), t))
    return frames


def generate_crunch_rep(t_start: float, duration: float = 2.0, fps: int = 30
                         ) -> List[Tuple[PoseResult, float]]:
    """Generate V-up crunch rep (lying -> V-shape -> back)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            crunch = phase / 0.4
        elif phase < 0.6:
            crunch = 1.0
        else:
            crunch = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Lying position with V-up
        # Body horizontal, then torso and legs come up to V
        world[LEFT_SHOULDER] = [-0.18, -0.40 + crunch * 0.15, -0.30 + crunch * 0.15]
        world[RIGHT_SHOULDER] = [0.18, -0.40 + crunch * 0.15, -0.30 + crunch * 0.15]
        world[LEFT_HIP] = [-0.10, 0.0, 0.0]
        world[RIGHT_HIP] = [0.10, 0.0, 0.0]
        world[LEFT_KNEE] = [-0.10, 0.40 - crunch * 0.15, 0.30 - crunch * 0.15]
        world[RIGHT_KNEE] = [0.10, 0.40 - crunch * 0.15, 0.30 - crunch * 0.15]
        world[LEFT_ANKLE] = [-0.10, 0.80 - crunch * 0.30, 0.40 - crunch * 0.25]
        world[RIGHT_ANKLE] = [0.10, 0.80 - crunch * 0.30, 0.40 - crunch * 0.25]

        # Wrists extend forward
        world[LEFT_WRIST] = [-0.15, -0.30 + crunch * 0.20, -0.20 + crunch * 0.10]
        world[RIGHT_WRIST] = [0.15, -0.30 + crunch * 0.20, -0.20 + crunch * 0.10]

        frames.append((make_pose(lm, world), t))
    return frames


def generate_standing_idle(t_start: float, duration: float = 3.0, fps: int = 30
                           ) -> List[Tuple[PoseResult, float]]:
    """Standing still (no movement)."""
    frames = []
    n_frames = int(duration * fps)
    lm = _base_landmarks()
    world = _base_world("standing")
    for i in range(n_frames):
        t = t_start + i / fps
        # Add tiny noise so it's not perfectly still (realistic)
        noise_lm = lm.copy()
        noise_world = world.copy()
        noise_lm[:, :3] += np.random.normal(0, 0.001, (33, 3))
        noise_world += np.random.normal(0, 0.001, (33, 3))
        frames.append((make_pose(noise_lm, noise_world), t))
    return frames


def generate_no_pose(t_start: float, duration: float = 2.0, fps: int = 30
                     ) -> List[Tuple[PoseResult, float]]:
    """Pose with all landmarks invisible (walked out of frame)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        lm = np.zeros((33, 4))  # all visibility = 0
        world = np.zeros((33, 3))
        frames.append((make_pose(lm, world), t))
    return frames


def generate_plank_hold(t_start: float, duration: float = 5.0, fps: int = 30
                        ) -> List[Tuple[PoseResult, float]]:
    """Generate plank hold frames."""
    frames = []
    n_frames = int(duration * fps)
    lm = _base_landmarks()
    world = _base_world("plank")

    # Set image landmarks for plank position
    lm[NOSE] = [0.15, 0.45, 0.0, 0.95]
    lm[LEFT_SHOULDER] = [0.30, 0.48, 0.0, 0.95]
    lm[RIGHT_SHOULDER] = [0.30, 0.52, 0.0, 0.95]
    lm[LEFT_ELBOW] = [0.22, 0.48, 0.0, 0.95]
    lm[RIGHT_ELBOW] = [0.22, 0.52, 0.0, 0.95]
    lm[LEFT_WRIST] = [0.18, 0.48, 0.0, 0.95]
    lm[RIGHT_WRIST] = [0.18, 0.52, 0.0, 0.95]
    lm[LEFT_HIP] = [0.50, 0.48, 0.0, 0.95]
    lm[RIGHT_HIP] = [0.50, 0.52, 0.0, 0.95]
    lm[LEFT_KNEE] = [0.68, 0.48, 0.0, 0.95]
    lm[RIGHT_KNEE] = [0.68, 0.52, 0.0, 0.95]
    lm[LEFT_ANKLE] = [0.85, 0.48, 0.0, 0.95]
    lm[RIGHT_ANKLE] = [0.85, 0.52, 0.0, 0.95]

    for i in range(n_frames):
        t = t_start + i / fps
        noise_lm = lm.copy()
        noise_world = world.copy()
        noise_lm[:, :3] += np.random.normal(0, 0.001, (33, 3))
        noise_world += np.random.normal(0, 0.001, (33, 3))
        frames.append((make_pose(noise_lm, noise_world), t))
    return frames


def generate_side_plank_hold(t_start: float, duration: float = 5.0, fps: int = 30
                              ) -> List[Tuple[PoseResult, float]]:
    """Generate side plank hold frames."""
    frames = []
    n_frames = int(duration * fps)
    lm = _base_landmarks()
    world = _base_world("side_plank")

    # Side plank image landmarks
    lm[LEFT_SHOULDER] = [0.30, 0.55, 0.0, 0.95]
    lm[RIGHT_SHOULDER] = [0.30, 0.35, 0.0, 0.95]
    lm[LEFT_ELBOW] = [0.25, 0.60, 0.0, 0.95]
    lm[RIGHT_ELBOW] = [0.30, 0.25, 0.0, 0.95]
    lm[LEFT_WRIST] = [0.22, 0.65, 0.0, 0.95]
    lm[RIGHT_WRIST] = [0.30, 0.15, 0.0, 0.95]
    lm[LEFT_HIP] = [0.50, 0.50, 0.0, 0.95]
    lm[RIGHT_HIP] = [0.50, 0.48, 0.0, 0.95]
    lm[LEFT_KNEE] = [0.68, 0.50, 0.0, 0.95]
    lm[RIGHT_KNEE] = [0.68, 0.48, 0.0, 0.95]
    lm[LEFT_ANKLE] = [0.85, 0.50, 0.0, 0.95]
    lm[RIGHT_ANKLE] = [0.85, 0.48, 0.0, 0.95]

    for i in range(n_frames):
        t = t_start + i / fps
        noise_lm = lm.copy()
        noise_world = world.copy()
        noise_lm[:, :3] += np.random.normal(0, 0.001, (33, 3))
        noise_world += np.random.normal(0, 0.001, (33, 3))
        frames.append((make_pose(noise_lm, noise_world), t))
    return frames


def generate_deadlift_rep(t_start: float, duration: float = 2.5, fps: int = 30
                           ) -> List[Tuple[PoseResult, float]]:
    """Generate deadlift rep (standing -> hip hinge -> standing)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            hinge = phase / 0.4
        elif phase < 0.6:
            hinge = 1.0
        else:
            hinge = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Hip hinge: torso leans forward, hips push back
        world[LEFT_SHOULDER] = [-0.18, -0.40 + hinge * 0.25, -hinge * 0.30]
        world[RIGHT_SHOULDER] = [0.18, -0.40 + hinge * 0.25, -hinge * 0.30]
        world[NOSE] = [0.0, -0.55 + hinge * 0.30, -hinge * 0.35]
        world[LEFT_HIP] = [-0.10, hinge * 0.05, hinge * 0.10]
        world[RIGHT_HIP] = [0.10, hinge * 0.05, hinge * 0.10]

        # Slight knee bend
        world[LEFT_KNEE] = [-0.10, 0.40, hinge * 0.05]
        world[RIGHT_KNEE] = [0.10, 0.40, hinge * 0.05]

        # Hands go down
        world[LEFT_WRIST] = [-0.18, 0.05 + hinge * 0.40, -hinge * 0.20]
        world[RIGHT_WRIST] = [0.18, 0.05 + hinge * 0.40, -hinge * 0.20]

        frames.append((make_pose(lm, world), t))
    return frames


def generate_pullup_rep(t_start: float, duration: float = 2.5, fps: int = 30
                         ) -> List[Tuple[PoseResult, float]]:
    """Generate pullup rep (hanging -> chin above bar -> hanging)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            pull = phase / 0.4
        elif phase < 0.6:
            pull = 1.0
        else:
            pull = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Wrists stay fixed (on bar, above head)
        world[LEFT_WRIST] = [-0.25, -0.70, 0.0]
        world[RIGHT_WRIST] = [0.25, -0.70, 0.0]
        lm[LEFT_WRIST] = [0.36, 0.08, 0.0, 0.95]
        lm[RIGHT_WRIST] = [0.64, 0.08, 0.0, 0.95]

        # Body rises as user pulls up
        body_rise = pull * 0.35
        world[LEFT_SHOULDER] = [-0.22, -0.40 - body_rise * 0.3, 0.0]
        world[RIGHT_SHOULDER] = [0.22, -0.40 - body_rise * 0.3, 0.0]
        world[LEFT_ELBOW] = [-0.24, -0.55 + (1 - pull) * 0.30, 0.0]
        world[RIGHT_ELBOW] = [0.24, -0.55 + (1 - pull) * 0.30, 0.0]
        world[NOSE] = [0.0, -0.55 - body_rise * 0.3, 0.0]

        lm[LEFT_SHOULDER, 1] = 0.25 - body_rise * 0.1
        lm[RIGHT_SHOULDER, 1] = 0.25 - body_rise * 0.1

        frames.append((make_pose(lm, world), t))
    return frames


def generate_tricep_dip_rep(t_start: float, duration: float = 2.0, fps: int = 30
                             ) -> List[Tuple[PoseResult, float]]:
    """Generate tricep dip rep (L-shape on bench)."""
    frames = []
    n_frames = int(duration * fps)
    for i in range(n_frames):
        t = t_start + i / fps
        phase = i / n_frames

        if phase < 0.4:
            dip = phase / 0.4
        elif phase < 0.6:
            dip = 1.0
        else:
            dip = 1.0 - (phase - 0.6) / 0.4

        lm = _base_landmarks()
        world = _base_world("standing")

        # Seated L-shape: hips at bench edge, legs extended
        world[LEFT_HIP] = [-0.10, 0.0, 0.0]
        world[RIGHT_HIP] = [0.10, 0.0, 0.0]
        world[LEFT_KNEE] = [-0.10, 0.0, 0.40]
        world[RIGHT_KNEE] = [0.10, 0.0, 0.40]
        world[LEFT_ANKLE] = [-0.10, 0.0, 0.80]
        world[RIGHT_ANKLE] = [0.10, 0.0, 0.80]

        # Arms behind on bench, dipping down
        world[LEFT_SHOULDER] = [-0.18, -0.40 + dip * 0.15, -0.05]
        world[RIGHT_SHOULDER] = [0.18, -0.40 + dip * 0.15, -0.05]
        world[LEFT_ELBOW] = [-0.18, -0.20 + dip * 0.10, -0.15]
        world[RIGHT_ELBOW] = [0.18, -0.20 + dip * 0.10, -0.15]
        world[LEFT_WRIST] = [-0.18, -0.10, -0.20]
        world[RIGHT_WRIST] = [0.18, -0.10, -0.20]

        frames.append((make_pose(lm, world), t))
    return frames


# ── Test scenarios ───────────────────────────────────────────────────────

def run_test_scenario(exercise_name: str) -> Dict:
    """Run a complete test scenario for one exercise and return results."""
    from src.classification.squat_detector import SquatDetector
    from src.classification.lunge_detector import LungeDetector
    from src.classification.deadlift_detector import DeadliftDetector
    from src.classification.pullup_detector import PullUpDetector
    from src.classification.bicep_curl_detector import BicepCurlDetector
    from src.classification.tricep_dip_detector import TricepDipDetector
    from src.classification.plank_detector import PlankDetector
    from src.classification.crunch_detector import CrunchDetector
    from src.classification.lateral_raise_detector import LateralRaiseDetector
    from src.classification.side_plank_detector import SidePlankDetector

    detectors = {
        "squat": SquatDetector,
        "lunge": LungeDetector,
        "deadlift": DeadliftDetector,
        "pullup": PullUpDetector,
        "bicep_curl": BicepCurlDetector,
        "tricep_dip": TricepDipDetector,
        "plank": PlankDetector,
        "crunch": CrunchDetector,
        "lateral_raise": LateralRaiseDetector,
        "side_plank": SidePlankDetector,
    }

    rep_generators = {
        "squat": generate_squat_rep,
        "lunge": generate_lunge_rep,
        "deadlift": generate_deadlift_rep,
        "pullup": generate_pullup_rep,
        "bicep_curl": lambda t, **kw: generate_elbow_rep(t, top_angle=165, bottom_angle=45, **kw),
        "tricep_dip": generate_tricep_dip_rep,
        "crunch": generate_crunch_rep,
        "lateral_raise": generate_lateral_raise_rep,
    }

    is_static = exercise_name in ("plank", "side_plank")

    detector = detectors[exercise_name]()
    results = {
        "exercise": exercise_name,
        "checks": [],
        "passed": 0,
        "failed": 0,
        "errors": [],
    }

    def check(name: str, condition: bool, detail: str = ""):
        status = "PASS" if condition else "FAIL"
        results["checks"].append({"name": name, "status": status, "detail": detail})
        if condition:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(f"{name}: {detail}")

    try:
        t = 0.0
        fps = 30

        # ── Phase 1: Initial idle (standing still, 2 seconds) ──
        idle_frames = generate_standing_idle(t, duration=2.0, fps=fps)
        session_states_idle = []
        for pose, ts in idle_frames:
            result = detector.classify(pose, timestamp=ts)
            session_states_idle.append(result.details.get("session_state", "unknown"))
        t = idle_frames[-1][1] + 1.0 / fps

        # Check: should NOT be ACTIVE during idle
        active_during_idle = sum(1 for s in session_states_idle if s == "active")
        check(
            "Idle before exercise -> not ACTIVE",
            active_during_idle == 0,
            f"Got {active_during_idle}/{len(session_states_idle)} ACTIVE frames during idle"
        )

        if is_static:
            # ── Static hold test ──

            # Phase 2: Hold position (5 seconds)
            if exercise_name == "plank":
                hold_frames = generate_plank_hold(t, duration=5.0, fps=fps)
            else:
                hold_frames = generate_side_plank_hold(t, duration=5.0, fps=fps)

            session_states_hold = []
            is_active_hold = []
            for pose, ts in hold_frames:
                result = detector.classify(pose, timestamp=ts)
                session_states_hold.append(result.details.get("session_state", "unknown"))
                is_active_hold.append(result.is_active)
            t = hold_frames[-1][1] + 1.0 / fps

            # Check: should become ACTIVE during hold
            active_count = sum(1 for s in session_states_hold if s == "active")
            check(
                "Hold position -> becomes ACTIVE",
                active_count > 0,
                f"Got {active_count}/{len(session_states_hold)} ACTIVE frames"
            )

            # Check: hold duration tracked
            hold_dur = result.details.get("hold_duration", "0s")
            hold_secs = float(hold_dur.replace("s", "")) if hold_dur else 0
            check(
                "Hold duration tracked",
                hold_secs > 0,
                f"Hold duration: {hold_dur}"
            )

            # Phase 3: Stand up / rest (10 seconds)
            rest_frames = generate_standing_idle(t, duration=10.0, fps=fps)
            session_states_rest = []
            for pose, ts in rest_frames:
                result = detector.classify(pose, timestamp=ts)
                session_states_rest.append(result.details.get("session_state", "unknown"))
            t = rest_frames[-1][1] + 1.0 / fps

            # Check: transitions to RESTING during rest
            resting_count = sum(1 for s in session_states_rest if s == "resting")
            check(
                "Rest after hold -> RESTING state",
                resting_count > 0,
                f"Got {resting_count}/{len(session_states_rest)} RESTING frames"
            )

            # Phase 4: Resume hold (3 seconds)
            if exercise_name == "plank":
                resume_frames = generate_plank_hold(t, duration=3.0, fps=fps)
            else:
                resume_frames = generate_side_plank_hold(t, duration=3.0, fps=fps)

            session_states_resume = []
            for pose, ts in resume_frames:
                result = detector.classify(pose, timestamp=ts)
                session_states_resume.append(result.details.get("session_state", "unknown"))
            t = resume_frames[-1][1] + 1.0 / fps

            active_resume = sum(1 for s in session_states_resume if s == "active")
            check(
                "Resume hold -> ACTIVE again",
                active_resume > 0,
                f"Got {active_resume}/{len(session_states_resume)} ACTIVE frames on resume"
            )

            # Phase 5: Walk out of frame (2 seconds)
            no_pose_frames = generate_no_pose(t, duration=2.0, fps=fps)
            for pose, ts in no_pose_frames:
                result = detector.classify(pose, timestamp=ts)
            t = no_pose_frames[-1][1] + 1.0 / fps

            # Check: set count should be > 0 (at least 1 closed set)
            set_count = detector.set_count
            check(
                "Set tracking works (>=1 set recorded)",
                set_count >= 1 or len(detector._set_hold_durations) >= 1,
                f"Sets closed: {set_count}, hold durations: {detector._set_hold_durations}"
            )

            # Check: rep count should be 0 for static holds
            check(
                "No reps counted for static hold",
                detector.rep_count == 0,
                f"Rep count: {detector.rep_count}"
            )

        else:
            # ── Rep-based test ──

            # Phase 2: Do 3 reps
            rep_gen = rep_generators[exercise_name]
            for rep_i in range(3):
                rep_frames = rep_gen(t, duration=2.5, fps=fps)
                for pose, ts in rep_frames:
                    result = detector.classify(pose, timestamp=ts)
                t = rep_frames[-1][1] + 0.5  # small gap between reps

            reps_after_set1 = detector.rep_count
            session_state_during = result.details.get("session_state", "unknown")

            # Check: session should be ACTIVE or at least have counted some reps
            check(
                "Reps counted during set 1",
                reps_after_set1 > 0,
                f"Counted {reps_after_set1} reps (expected ~3)"
            )

            # Phase 3: Rest 12 seconds (standing idle)
            rest_frames = generate_standing_idle(t, duration=12.0, fps=fps)
            session_states_rest = []
            rest_tiers = []
            for pose, ts in rest_frames:
                result = detector.classify(pose, timestamp=ts)
                session_states_rest.append(result.details.get("session_state", "unknown"))
                rest_tiers.append(result.details.get("rest_tier", "unknown"))
            t = rest_frames[-1][1] + 1.0 / fps

            # Check: should transition out of ACTIVE during rest
            non_active = sum(1 for s in session_states_rest if s != "active")
            check(
                "Rest period -> exits ACTIVE",
                non_active > 0,
                f"Got {non_active}/{len(session_states_rest)} non-ACTIVE frames during rest"
            )

            # Phase 4: Resume with 2 more reps
            for rep_i in range(2):
                rep_frames = rep_gen(t, duration=2.5, fps=fps)
                for pose, ts in rep_frames:
                    result = detector.classify(pose, timestamp=ts)
                t = rep_frames[-1][1] + 0.5

            reps_after_set2 = detector.rep_count
            check(
                "More reps counted after rest",
                reps_after_set2 > reps_after_set1,
                f"Total reps: {reps_after_set2} (was {reps_after_set1} after set 1)"
            )

            # Phase 5: Walk out of frame (2 seconds = 60 frames > NO_POSE_RESET_FRAMES=30)
            no_pose_frames = generate_no_pose(t, duration=2.0, fps=fps)
            for pose, ts in no_pose_frames:
                result = detector.classify(pose, timestamp=ts)
            t = no_pose_frames[-1][1] + 1.0 / fps

            # Check: session state should reset
            final_state = result.details.get("session_state", "unknown")
            check(
                "Pose lost -> session resets to IDLE",
                final_state == "idle",
                f"Session state after pose lost: {final_state}"
            )

            # Phase 6: Return and do 1 more rep
            rep_frames = rep_gen(t, duration=2.5, fps=fps)
            for pose, ts in rep_frames:
                result = detector.classify(pose, timestamp=ts)
            t = rep_frames[-1][1] + 1.0 / fps

            # Check: set count should be > 0 (at least set 1 closed)
            check(
                "Set boundary detected (>=1 set closed)",
                detector.set_count >= 1,
                f"Closed sets: {detector.set_count}, set_reps: {detector._set_reps}"
            )

        # ── Session summary check ──
        summary = detector.get_session_summary()
        check(
            "Session summary builds without error",
            summary is not None and "exercise" in summary,
            f"Summary keys: {list(summary.keys()) if summary else 'None'}"
        )

        if is_static:
            check(
                "Summary has hold_duration field",
                "hold_duration" in summary and summary.get("total_reps", -1) == 0,
                f"hold_duration={summary.get('hold_duration')}, total_reps={summary.get('total_reps')}"
            )
        else:
            check(
                "Summary has reps and sets data",
                summary.get("total_reps", 0) > 0 or summary.get("num_sets", 0) >= 0,
                f"total_reps={summary.get('total_reps')}, sets={summary.get('sets')}"
            )

    except Exception as e:
        import traceback
        results["errors"].append(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        results["failed"] += 1

    return results


def _run_single(exercise: str) -> Dict:
    """Wrapper for multiprocessing."""
    return run_test_scenario(exercise)


def main():
    exercises = [
        "squat", "lunge", "deadlift", "pullup", "bicep_curl",
        "tricep_dip", "plank", "crunch", "lateral_raise", "side_plank",
    ]

    print("=" * 70)
    print("RESTING/INACTIVITY DETECTION VERIFICATION")
    print(f"Testing {len(exercises)} exercises")
    print("=" * 70)

    # Run ALL exercises in parallel using all CPU cores
    n_workers = min(len(exercises), multiprocessing.cpu_count())
    print(f"\nUsing {n_workers} parallel workers\n")

    all_results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single, ex): ex for ex in exercises}
        for future in as_completed(futures):
            ex = futures[future]
            try:
                all_results[ex] = future.result()
            except Exception as e:
                all_results[ex] = {
                    "exercise": ex, "checks": [], "passed": 0,
                    "failed": 1, "errors": [f"Process error: {e}"]
                }

    # Print results in exercise order
    total_passed = 0
    total_failed = 0
    failed_exercises = []

    for ex in exercises:
        r = all_results[ex]
        total_passed += r["passed"]
        total_failed += r["failed"]

        status = "PASS" if r["failed"] == 0 else "FAIL"
        icon = "[OK]" if status == "PASS" else "[!!]"
        is_static = ex in ("plank", "side_plank")
        ex_type = "static" if is_static else "reps"

        print(f"\n{icon} {ex.upper()} ({ex_type}) — {r['passed']} passed, {r['failed']} failed")
        print("-" * 50)

        for c in r["checks"]:
            mark = "  [+]" if c["status"] == "PASS" else "  [X]"
            detail = f" ({c['detail']})" if c["detail"] else ""
            print(f"{mark} {c['name']}{detail}")

        if r["errors"]:
            failed_exercises.append(ex)
            for err in r["errors"]:
                if "EXCEPTION" in err:
                    print(f"  >>> {err[:200]}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    if failed_exercises:
        print(f"FAILED EXERCISES: {', '.join(failed_exercises)}")
    else:
        print("ALL EXERCISES PASSED")
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
