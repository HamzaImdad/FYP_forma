"""
Maps classification results to visual feedback (joint colors, skeleton colors, text).
"""

from typing import Dict, List, Tuple

from ..classification.base import ClassificationResult
from ..utils.constants import LANDMARK_INDICES, SKELETON_CONNECTIONS

# BGR colors for OpenCV rendering
COLORS = {
    "correct":   (0, 255, 0),     # Green
    "incorrect": (0, 0, 255),     # Red
    "warning":   (0, 165, 255),   # Orange
    "neutral":   (200, 200, 200), # Gray
}


class FeedbackEngine:
    """Generates visual feedback from classification results."""

    def get_joint_colors(self, result: ClassificationResult) -> Dict[int, Tuple[int, int, int]]:
        """
        Map landmark indices to BGR colors based on joint feedback.
        Returns dict of landmark_index -> (B, G, R).
        """
        colors = {}
        if not result.joint_feedback:
            return colors
        for joint_name, status in result.joint_feedback.items():
            if joint_name in LANDMARK_INDICES:
                idx = LANDMARK_INDICES[joint_name]
                colors[idx] = COLORS.get(status, COLORS["neutral"])
        return colors

    def get_skeleton_colors(
        self, result: ClassificationResult
    ) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """
        Get color for each skeleton connection.
        Returns list of (idx_a, idx_b, color) tuples.
        """
        joint_colors = self.get_joint_colors(result)
        skeleton = []

        for idx_a, idx_b in SKELETON_CONNECTIONS:
            color_a = joint_colors.get(idx_a, COLORS["neutral"])
            color_b = joint_colors.get(idx_b, COLORS["neutral"])

            if color_a == COLORS["incorrect"] or color_b == COLORS["incorrect"]:
                color = COLORS["incorrect"]
            elif color_a == COLORS["warning"] or color_b == COLORS["warning"]:
                color = COLORS["warning"]
            elif color_a == COLORS["correct"] and color_b == COLORS["correct"]:
                color = COLORS["correct"]
            else:
                color = COLORS["neutral"]

            skeleton.append((idx_a, idx_b, color))

        return skeleton

    def get_text_feedback(self, result: ClassificationResult) -> str:
        """Generate human-readable text feedback."""
        conf = result.confidence if result.confidence is not None else 0.0
        if result.is_correct:
            return f"Good form! ({conf:.0%} confidence)"

        issues = list(result.details.values()) if result.details else []
        msg = f"Form issues detected ({conf:.0%}):\n"
        msg += "\n".join(f"  - {issue}" for issue in issues[:3])
        return msg
