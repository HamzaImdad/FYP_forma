"""
Per-exercise feature definitions.

Each exercise specifies:
  - primary_angles: Joint angles that directly determine correct form
  - secondary_angles: Supporting angles to check
  - custom_features: Additional computed features (symmetry, alignment, etc.)
  - temporal_features: Motion features over frame sequences
    Naming: "{angle}_rom", "{landmark}_velocity", "{landmark}_accel"
"""

EXERCISE_FEATURES = {
    "squat": {
        "primary_angles": ["left_knee", "right_knee", "left_hip", "right_hip"],
        "secondary_angles": ["left_ankle", "right_ankle"],
        "custom_features": [
            "knee_symmetry",
            "hip_symmetry",
            "torso_lean",
            "knee_over_toe",
            "hip_below_knee",
        ],
        "temporal_features": [
            "left_knee_rom",
            "right_knee_rom",
            "left_hip_rom",
            "left_hip_velocity",
            "left_knee_velocity",
        ],
    },
    "lunge": {
        "primary_angles": ["left_knee", "right_knee", "left_hip", "right_hip"],
        "secondary_angles": ["left_ankle", "right_ankle"],
        "custom_features": [
            "knee_symmetry",
            "torso_lean",
            "hip_alignment",
        ],
        "temporal_features": [
            "left_knee_rom",
            "right_knee_rom",
            "left_hip_velocity",
        ],
    },
    "deadlift": {
        "primary_angles": ["left_hip", "right_hip", "left_knee", "right_knee"],
        "secondary_angles": ["left_shoulder", "right_shoulder"],
        "custom_features": [
            "spine_alignment",
            "hip_symmetry",
            "back_straightness",
        ],
        "temporal_features": [
            "left_hip_rom",
            "right_hip_rom",
            "left_hip_velocity",
            "left_wrist_velocity",
        ],
    },
    "bench_press": {
        "primary_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "secondary_angles": [],
        "custom_features": [
            "elbow_symmetry",
            "shoulder_symmetry",
            "wrist_over_elbow",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_wrist_velocity",
        ],
    },
    "overhead_press": {
        "primary_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "secondary_angles": ["left_hip", "right_hip"],
        "custom_features": [
            "elbow_symmetry",
            "torso_lean",
            "lockout_angle",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_wrist_velocity",
        ],
    },
    "pullup": {
        "primary_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "secondary_angles": [],
        "custom_features": [
            "elbow_symmetry",
            "chin_above_bar",
            "body_swing",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_wrist_velocity",
            "left_wrist_accel",
        ],
    },
    "pushup": {
        "primary_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "secondary_angles": ["left_hip", "right_hip"],
        "custom_features": [
            "elbow_symmetry",
            "body_line",
            "hip_sag",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_shoulder_velocity",
        ],
    },
    "plank": {
        "primary_angles": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
        "secondary_angles": ["left_knee", "right_knee"],
        "custom_features": [
            "body_line",
            "hip_sag",
        ],
        "temporal_features": [
            "left_hip_velocity",
            "left_shoulder_velocity",
        ],
    },
    "bicep_curl": {
        "primary_angles": ["left_elbow", "right_elbow"],
        "secondary_angles": ["left_shoulder", "right_shoulder"],
        "custom_features": [
            "elbow_symmetry",
            "upper_arm_movement",
            "torso_swing",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_wrist_velocity",
            "left_wrist_accel",
        ],
    },
    "tricep_dip": {
        "primary_angles": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "secondary_angles": [],
        "custom_features": [
            "elbow_symmetry",
            "forward_lean",
        ],
        "temporal_features": [
            "left_elbow_rom",
            "right_elbow_rom",
            "left_wrist_velocity",
        ],
    },
}
