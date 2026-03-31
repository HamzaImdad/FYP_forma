"""
Tests for build_sequences.py manifest-based sequence builder.

Covers:
- No aug_ video_ids in output .npz
- Temporal order preserved (original CSV row order)
- Training stride = seq_len (non-overlapping)
- Val/test stride = seq_len // 2
- No balanced-CSV path (build_balanced_landmark_sequences_for_exercise disabled)
- Output per-partition .npz naming convention
- balance_data.py augment_landmark_minority gated off for BiLSTM pipeline
"""

import sys
import textwrap
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import LANDMARK_NAMES, NUM_LANDMARKS


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_landmark_row(frame_idx: int, label: str = "correct") -> dict:
    """Create a single landmark CSV row with identifiable frame_idx values.

    We embed frame_idx into the first landmark x-coordinate so we can verify
    temporal order without needing real pose data.
    """
    row = {}
    for name in LANDMARK_NAMES:
        row[f"lm_{name}_x"] = 0.0
        row[f"lm_{name}_y"] = 0.0
        row[f"lm_{name}_z"] = 0.0

    # Embed frame_idx in the first landmark x so we can detect order changes
    row[f"lm_{LANDMARK_NAMES[0]}_x"] = float(frame_idx)

    # Add world landmark columns expected by row_to_pose_result
    for name in LANDMARK_NAMES:
        row[f"{name}_x"] = 0.01
        row[f"{name}_y"] = 0.01
        row[f"{name}_z"] = 0.01
        row[f"{name}_vis"] = 1.0
        row[f"{name}_wx"] = 0.01
        row[f"{name}_wy"] = 0.01
        row[f"{name}_wz"] = 0.01

    row["label"] = label
    row["timestamp_ms"] = frame_idx * 33

    return row


def _write_fake_csv(path: Path, n_frames: int = 60, label: str = "correct") -> None:
    """Write a fake landmark CSV with n_frames rows in sequential frame order."""
    rows = [_make_landmark_row(i, label) for i in range(n_frames)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _setup_fake_data(
    tmp_path: Path,
    exercise: str = "squat",
    n_videos: int = 3,
    frames_per_video: int = 60,
    label: str = "correct",
):
    """Create fake landmark CSVs and a manifest for one partition.

    Returns (landmarks_dir, splits_dir, csv_names_list).
    """
    landmarks_dir = tmp_path / "landmarks"
    landmarks_dir.mkdir(parents=True)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)

    csv_names = []
    for i in range(n_videos):
        csv_name = f"youtube_{exercise}_{label}_vid{i:03d}.csv"
        _write_fake_csv(landmarks_dir / csv_name, n_frames=frames_per_video, label=label)
        csv_names.append(csv_name)

    return landmarks_dir, splits_dir, csv_names


# ---------------------------------------------------------------------------
# Import target under test
# ---------------------------------------------------------------------------

def _import_build_partition_sequences():
    """Import build_partition_sequences from scripts/build_sequences.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_sequences",
        str(PROJECT_ROOT / "scripts" / "build_sequences.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildPartitionSequences:
    """Tests for the new manifest-based build_partition_sequences function."""

    def test_output_per_partition_naming(self, tmp_path):
        """Output files must follow {exercise}_{partition}_sequences.npz naming."""
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(tmp_path, exercise="squat")
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_train.txt"
        manifest.write_text("\n".join(csv_names) + "\n")

        # Import the module
        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        expected = sequences_dir / "squat_train_sequences.npz"
        assert expected.exists(), f"Expected output file {expected} not found"

    def test_no_aug_video_ids_in_npz(self, tmp_path):
        """No video_id in the output .npz should start with 'aug_'."""
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(tmp_path, exercise="squat")
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        # Add a fake aug_ CSV that should be excluded from the manifest
        aug_csv = landmarks_dir / "aug_youtube_squat_correct_fake.csv"
        _write_fake_csv(aug_csv, n_frames=60)

        # Manifest contains only the real CSVs, not the aug_ one
        manifest = splits_dir / "squat_train.txt"
        manifest.write_text("\n".join(csv_names) + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        npz_path = sequences_dir / "squat_train_sequences.npz"
        data = np.load(npz_path, allow_pickle=True)
        video_ids = data["video_ids"]

        aug_ids = [v for v in video_ids if str(v).startswith("aug_")]
        assert len(aug_ids) == 0, f"Found aug_ video_ids: {aug_ids}"

    def test_training_stride_non_overlapping(self, tmp_path):
        """For training partition, stride must equal seq_len (no overlap).

        With 60 frames and seq_len=30, stride=30, we expect exactly 2 non-overlapping
        windows per video: [0..29] and [30..59].
        """
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(
            tmp_path, exercise="squat", n_videos=1, frames_per_video=60
        )
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_train.txt"
        manifest.write_text(csv_names[0] + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_train_sequences.npz", allow_pickle=True)
        X = data["X"]
        # 60 frames, stride=30, seq_len=30 → 2 windows (not 3 or 4)
        assert X.shape[0] == 2, f"Expected 2 non-overlapping windows, got {X.shape[0]}"

    def test_val_stride_half(self, tmp_path):
        """For val partition, stride must equal seq_len // 2 (50% overlap).

        With 60 frames and seq_len=30, stride=15, we expect:
        windows at 0, 15, 30 → 3 windows.
        """
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(
            tmp_path, exercise="squat", n_videos=1, frames_per_video=60
        )
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_val.txt"
        manifest.write_text(csv_names[0] + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="val",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_val_sequences.npz", allow_pickle=True)
        X = data["X"]
        # 60 frames, stride=15, seq_len=30 → 3 windows
        assert X.shape[0] == 3, f"Expected 3 overlapping windows (stride=15), got {X.shape[0]}"

    def test_test_stride_half(self, tmp_path):
        """For test partition, stride must equal seq_len // 2."""
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(
            tmp_path, exercise="squat", n_videos=1, frames_per_video=60
        )
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_test.txt"
        manifest.write_text(csv_names[0] + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="test",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_test_sequences.npz", allow_pickle=True)
        X = data["X"]
        assert X.shape[0] == 3, f"Expected 3 overlapping windows (stride=15), got {X.shape[0]}"

    def test_temporal_order_preserved(self, tmp_path):
        """Features within each window must be in original CSV row order.

        We embed frame_idx into the first landmark x-coordinate and verify that
        within each sequence window the values are monotonically increasing.
        """
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(
            tmp_path, exercise="squat", n_videos=1, frames_per_video=60
        )
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_train.txt"
        manifest.write_text(csv_names[0] + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_train_sequences.npz", allow_pickle=True)
        X = data["X"]  # (n_windows, 30, 99)

        # Check that within each window, the first feature (lm_nose_x = frame_idx)
        # is monotonically non-decreasing
        for win_idx in range(X.shape[0]):
            frame_vals = X[win_idx, :, 0]  # first feature across 30 frames
            assert np.all(np.diff(frame_vals) >= 0), (
                f"Window {win_idx}: frame order not preserved. "
                f"Values: {frame_vals}"
            )

    def test_output_shape(self, tmp_path):
        """Output X must have shape (n, seq_len, 99)."""
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(
            tmp_path, exercise="squat", n_videos=2, frames_per_video=60
        )
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_train.txt"
        manifest.write_text("\n".join(csv_names) + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_train_sequences.npz", allow_pickle=True)
        X = data["X"]
        assert X.shape[1] == 30, f"Expected seq_len=30, got {X.shape[1]}"
        assert X.shape[2] == 99, f"Expected 99 features, got {X.shape[2]}"

    def test_npz_has_required_keys(self, tmp_path):
        """Output .npz must contain X, y, video_ids, feature_names keys."""
        landmarks_dir, splits_dir, csv_names = _setup_fake_data(tmp_path, exercise="squat")
        sequences_dir = tmp_path / "sequences"
        sequences_dir.mkdir()

        manifest = splits_dir / "squat_train.txt"
        manifest.write_text("\n".join(csv_names) + "\n")

        mod = _import_build_partition_sequences()
        mod.build_partition_sequences(
            exercise="squat",
            partition="train",
            splits_dir=splits_dir,
            landmarks_dir=landmarks_dir,
            sequences_dir=sequences_dir,
            seq_len=30,
        )

        data = np.load(sequences_dir / "squat_train_sequences.npz", allow_pickle=True)
        for key in ("X", "y", "video_ids", "feature_names"):
            assert key in data, f"Missing key '{key}' in output .npz"


class TestBuildBalancedDisabled:
    """Tests that the old balanced-CSV code path is no longer the default."""

    def test_no_offline_augmentation_in_sequences(self):
        """build_balanced_landmark_sequences_for_exercise must not be the default path.

        We check that the function either doesn't exist or is marked as
        disabled/removed in the source code.
        """
        source_path = PROJECT_ROOT / "scripts" / "build_sequences.py"
        source = source_path.read_text()

        # The function should either not exist in the source, or contain a
        # clear deprecation/removal marker
        has_function = "def build_balanced_landmark_sequences_for_exercise" in source

        if has_function:
            # Acceptable: function exists but is commented out or has removal marker
            assert (
                "# Removed" in source
                or "# REMOVED" in source
                or "# Deprecated" in source
                or "DEPRECATED" in source
                or "raise NotImplementedError" in source
            ), (
                "build_balanced_landmark_sequences_for_exercise still exists and "
                "appears to be active (no deprecation/removal marker found)"
            )


class TestBalanceDataAugmentationGated:
    """Tests that augment_landmark_minority is gated off in balance_data.py."""

    def test_augment_landmark_minority_gated(self):
        """augment_landmark_minority must be gated with deprecation warning.

        Reads the source of balance_data.py and verifies that the function
        body contains an early return and a comment indicating it is disabled
        for the BiLSTM pipeline.
        """
        source_path = PROJECT_ROOT / "scripts" / "balance_data.py"
        source = source_path.read_text()

        # Must contain a reference to why it's gated
        assert "DATA-05" in source or "SequenceDataset" in source or "offline frame" in source.lower(), (
            "augment_landmark_minority in balance_data.py is not gated: "
            "expected a comment referencing DATA-05 or SequenceDataset or 'offline frame'"
        )

        # Import the module and call the function — it must return unchanged df
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "balance_data",
            str(source_path),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Create a minimal DataFrame with required columns
        from src.feature_extraction.landmark_features import LANDMARK_FEATURE_NAMES
        data = {col: [0.1, 0.2] for col in LANDMARK_FEATURE_NAMES[:6]}
        data["exercise"] = ["squat", "squat"]
        data["label"] = ["correct", "correct"]
        data["source"] = ["youtube", "youtube"]
        data["video_id"] = ["vid001", "vid002"]
        df = pd.DataFrame(data)

        result = mod.augment_landmark_minority(df, target_n=100, exercise="squat")

        # Gated: must return the original df unchanged (no aug_ rows added)
        aug_rows = result[result["video_id"].str.startswith("aug_")]
        assert len(aug_rows) == 0, (
            f"augment_landmark_minority is not gated: it added {len(aug_rows)} aug_ rows "
            "when it should return the input DataFrame unchanged"
        )

    def test_augment_landmark_minority_prints_deprecation(self, capsys):
        """augment_landmark_minority must print a deprecation warning when called."""
        import importlib.util
        source_path = PROJECT_ROOT / "scripts" / "balance_data.py"
        spec = importlib.util.spec_from_file_location("balance_data", str(source_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from src.feature_extraction.landmark_features import LANDMARK_FEATURE_NAMES
        data = {col: [0.1] for col in LANDMARK_FEATURE_NAMES[:6]}
        data["exercise"] = ["squat"]
        data["label"] = ["correct"]
        data["source"] = ["youtube"]
        data["video_id"] = ["vid001"]
        df = pd.DataFrame(data)

        mod.augment_landmark_minority(df, target_n=100, exercise="squat")
        captured = capsys.readouterr()

        assert len(captured.out) > 0 or len(captured.err) > 0, (
            "augment_landmark_minority did not print any deprecation warning"
        )
