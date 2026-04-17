"""
Base classes for form classification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClassificationResult:
    """Result of form classification for a single frame/pose."""
    exercise: str
    is_correct: bool
    confidence: float                                # 0.0 to 1.0
    joint_feedback: Dict[str, str] = field(default_factory=dict)  # joint -> "correct"|"incorrect"|"warning"
    details: Dict[str, str] = field(default_factory=dict)         # human-readable feedback per issue
    is_active: bool = True                           # whether user is actively exercising
    form_score: float = 0.0                          # continuous 0.0-1.0 form quality score
    dtw_similarity: float = 1.0                      # rep-level template match in [0,1], 1.0 = no template/best
    dtw_worst_joint: Optional[str] = None            # joint group that deviated most (mnmDTW), when similarity is low


class FormClassifier(ABC):
    """Abstract base class for exercise form classifiers."""

    @abstractmethod
    def classify(self, features: Dict[str, Optional[float]], exercise: str) -> ClassificationResult:
        """Classify form based on extracted features."""
        pass

    @abstractmethod
    def get_supported_exercises(self) -> List[str]:
        """Return list of exercises this classifier supports."""
        pass
