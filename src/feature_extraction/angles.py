"""
Joint angle computation from pose landmarks.
"""

from typing import Dict, Optional

from ..pose_estimation.base import PoseResult
from ..utils.constants import JOINT_ANGLES, MIN_VISIBILITY
from ..utils.geometry import calculate_angle


def compute_joint_angles(
    pose: PoseResult,
    use_world: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Compute all defined joint angles from a PoseResult.

    Args:
        pose: PoseResult with landmarks.
        use_world: If True, use world coordinates (better for angle accuracy).

    Returns:
        Dict mapping angle name -> degrees (0-180), or None if landmarks not visible.
    """
    angles = {}

    for name, (idx_a, idx_b, idx_c) in JOINT_ANGLES.items():
        if not all(pose.is_visible(idx, MIN_VISIBILITY) for idx in [idx_a, idx_b, idx_c]):
            angles[name] = None
            continue

        if use_world:
            a = pose.get_world_landmark(idx_a)
            b = pose.get_world_landmark(idx_b)
            c = pose.get_world_landmark(idx_c)
        else:
            a = pose.get_landmark(idx_a)
            b = pose.get_landmark(idx_b)
            c = pose.get_landmark(idx_c)

        angles[name] = calculate_angle(a, b, c)

    return angles
