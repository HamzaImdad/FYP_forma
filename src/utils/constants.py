"""
Central definitions for exercises, landmarks, joint angles, and skeleton connections.
"""

# 10 target exercises (home-friendly; dropped bench_press, overhead_press, lunge)
EXERCISES = [
    "squat", "deadlift",
    "pullup", "pushup", "plank", "bicep_curl", "tricep_dip",
    "crunch", "lateral_raise", "side_plank",
]

# 33 MediaPipe BlazePose landmark names (index order)
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

NUM_LANDMARKS = 33

# Landmark name -> index lookup
LANDMARK_INDICES = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# Shorthand index constants for commonly used body landmarks
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Joint angle definitions: (point_a, vertex, point_b)
# Angle is measured at the vertex
JOINT_ANGLES = {
    "left_knee":      (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "right_knee":     (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
    "left_hip":       (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    "right_hip":      (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    "left_elbow":     (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "right_elbow":    (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "left_shoulder":  (LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    "right_shoulder": (RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
    "left_ankle":     (LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL),
    "right_ankle":    (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL),
}

# Skeleton connections for drawing overlays
SKELETON_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]

# Minimum visibility to consider a landmark valid
MIN_VISIBILITY = 0.5
