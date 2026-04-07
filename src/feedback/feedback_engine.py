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

    @staticmethod
    def score_to_color(form_score: float) -> Tuple[int, int, int]:
        """Map a 0.0-1.0 form score to a BGR color (red -> orange -> green)."""
        s = max(0.0, min(1.0, form_score))
        if s < 0.5:
            # Red (0,0,255) -> Orange (0,165,255)
            t = s / 0.5
            return (0, int(165 * t), 255)
        else:
            # Orange (0,165,255) -> Green (0,255,0)
            t = (s - 0.5) / 0.5
            return (0, int(165 + 90 * t), int(255 * (1 - t)))

    def get_skeleton_colors(
        self, result: ClassificationResult
    ) -> List[Tuple[int, int, Tuple[int, int, int]]]:
        """
        Get color for each skeleton connection.
        Uses per-joint colors where available, falls back to score-based overall color.
        Returns list of (idx_a, idx_b, color) tuples.
        """
        joint_colors = self.get_joint_colors(result)

        # Default color based on form_score when no per-joint info
        if not result.is_active:
            default_color = COLORS["neutral"]
        elif joint_colors:
            default_color = COLORS["neutral"]
        else:
            default_color = self.score_to_color(result.form_score)

        skeleton = []
        for idx_a, idx_b in SKELETON_CONNECTIONS:
            color_a = joint_colors.get(idx_a)
            color_b = joint_colors.get(idx_b)

            if color_a == COLORS["incorrect"] or color_b == COLORS["incorrect"]:
                color = COLORS["incorrect"]
            elif color_a == COLORS["warning"] or color_b == COLORS["warning"]:
                color = COLORS["warning"]
            elif color_a == COLORS["correct"] and color_b == COLORS["correct"]:
                color = COLORS["correct"]
            elif color_a is not None or color_b is not None:
                color = color_a or color_b
            else:
                color = default_color

            skeleton.append((idx_a, idx_b, color))

        return skeleton

    def get_text_feedback(self, result: ClassificationResult) -> str:
        """Generate human-readable text feedback based on form score."""
        # Inactive — user is not exercising
        if not result.is_active:
            return "Begin exercise movement..."

        score_pct = int(result.form_score * 100)
        issues = list(result.details.values()) if result.details else []

        if result.form_score > 0.7:
            return f"Good form! Score: {score_pct}/100"
        elif result.form_score >= 0.4:
            msg = f"Moderate form. Score: {score_pct}/100"
            if issues:
                msg += "\n" + "\n".join(f"  - {i}" for i in issues[:3])
            return msg
        else:
            msg = f"Needs improvement. Score: {score_pct}/100"
            if issues:
                msg += "\n" + "\n".join(f"  - {i}" for i in issues[:3])
            return msg
