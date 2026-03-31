"""
Unit tests for scripts/split_videos.py manifest generation.

Tests validate:
1. No aug_ prefixed filenames leak into any manifest
2. No video base ID appears in more than one partition (no data leakage)
3. .f398 format variants of the same video are grouped in the same partition
4. All 10 exercises produce train/val/test manifest files
5. Running split_videos.py twice without --force exits without overwriting
6. Each non-trivial partition has both correct and incorrect labels where possible
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root so we can import scripts/split_videos.py as a module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from split_videos import split_exercise_videos
from src.utils.constants import EXERCISES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fake_csv(directory: Path, name: str) -> Path:
    """Create an empty CSV file with the given name in directory."""
    p = directory / name
    p.write_text("frame\n0\n")
    return p


@pytest.fixture
def fake_landmarks_dir(tmp_path):
    """Landmarks directory with 10 fake CSVs for 'squat' exercise.

    5 correct + 5 incorrect source videos.
    Includes one .f398 variant pair (same base video, two format files).
    Includes one kaggle_images_ file (should be excluded).
    """
    lm_dir = tmp_path / "landmarks"
    lm_dir.mkdir()

    # 4 correct source videos (individual)
    make_fake_csv(lm_dir, "youtube_squat_correct_VID001.csv")
    make_fake_csv(lm_dir, "youtube_squat_correct_VID002.csv")
    make_fake_csv(lm_dir, "youtube_squat_correct_VID003.csv")
    make_fake_csv(lm_dir, "youtube_squat_correct_VID004.csv")

    # 1 correct source video that has a .f398 variant (two files, same base ID)
    make_fake_csv(lm_dir, "youtube_squat_correct_VID005.csv")
    make_fake_csv(lm_dir, "youtube_squat_correct_VID005.f398.csv")

    # 5 incorrect source videos
    make_fake_csv(lm_dir, "youtube_squat_incorrect_VID006.csv")
    make_fake_csv(lm_dir, "youtube_squat_incorrect_VID007.csv")
    make_fake_csv(lm_dir, "youtube_squat_incorrect_VID008.csv")
    make_fake_csv(lm_dir, "youtube_squat_incorrect_VID009.csv")
    make_fake_csv(lm_dir, "youtube_squat_incorrect_VID010.csv")

    # Kaggle image file — must be excluded from manifests
    make_fake_csv(lm_dir, "kaggle_images_squat_correct_00001.csv")

    # Augmented file — must NEVER appear in manifests
    make_fake_csv(lm_dir, "aug_youtube_squat_correct_VID001_0.csv")

    return lm_dir


@pytest.fixture
def splits_dir(tmp_path):
    """Output splits directory."""
    s = tmp_path / "splits"
    s.mkdir()
    return s


@pytest.fixture
def manifests(fake_landmarks_dir, splits_dir):
    """Run split_exercise_videos for 'squat' and return (train, val, test) file lists."""
    split_exercise_videos("squat", fake_landmarks_dir, splits_dir)

    train_file = splits_dir / "squat_train.txt"
    val_file = splits_dir / "squat_val.txt"
    test_file = splits_dir / "squat_test.txt"

    def read_manifest(path):
        if not path.exists():
            return []
        lines = path.read_text().strip().splitlines()
        return [ln.strip() for ln in lines if ln.strip()]

    train = read_manifest(train_file)
    val = read_manifest(val_file)
    test = read_manifest(test_file)
    return train, val, test


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoAugInManifests:
    """test_no_aug_in_manifests: No manifest file contains a filename starting with 'aug_'"""

    def test_train_has_no_aug(self, manifests):
        train, val, test = manifests
        aug_entries = [f for f in train if f.startswith("aug_")]
        assert aug_entries == [], f"Train manifest contains aug_ files: {aug_entries}"

    def test_val_has_no_aug(self, manifests):
        train, val, test = manifests
        aug_entries = [f for f in val if f.startswith("aug_")]
        assert aug_entries == [], f"Val manifest contains aug_ files: {aug_entries}"

    def test_test_has_no_aug(self, manifests):
        train, val, test = manifests
        aug_entries = [f for f in test if f.startswith("aug_")]
        assert aug_entries == [], f"Test manifest contains aug_ files: {aug_entries}"


class TestNoPartitionOverlap:
    """test_no_partition_overlap: No video base ID appears in more than one partition."""

    def get_base_id(self, filename: str) -> str:
        """Strip .f398 suffix and .csv extension for grouping."""
        stem = Path(filename).stem
        if stem.endswith(".f398"):
            stem = stem[:-5]
        return stem

    def test_train_val_no_overlap(self, manifests):
        train, val, test = manifests
        train_ids = {self.get_base_id(f) for f in train}
        val_ids = {self.get_base_id(f) for f in val}
        overlap = train_ids & val_ids
        assert overlap == set(), f"Train/Val overlap on base IDs: {overlap}"

    def test_train_test_no_overlap(self, manifests):
        train, val, test = manifests
        train_ids = {self.get_base_id(f) for f in train}
        test_ids = {self.get_base_id(f) for f in test}
        overlap = train_ids & test_ids
        assert overlap == set(), f"Train/Test overlap on base IDs: {overlap}"

    def test_val_test_no_overlap(self, manifests):
        train, val, test = manifests
        val_ids = {self.get_base_id(f) for f in val}
        test_ids = {self.get_base_id(f) for f in test}
        overlap = val_ids & test_ids
        assert overlap == set(), f"Val/Test overlap on base IDs: {overlap}"


class TestF398VariantsSamePartition:
    """test_f398_variants_same_partition: .f398 variants of same video end up in same partition."""

    def test_f398_variants_grouped(self, manifests):
        train, val, test = manifests
        all_partitions = {"train": train, "val": val, "test": test}

        base_vid = "youtube_squat_correct_VID005"
        f398_variant = "youtube_squat_correct_VID005.f398.csv"
        base_variant = "youtube_squat_correct_VID005.csv"

        # Find which partition each variant is in
        partition_of = {}
        for part_name, entries in all_partitions.items():
            for entry in entries:
                if entry in (f398_variant, base_variant):
                    partition_of[entry] = part_name

        # Both variants must be present and in the same partition
        assert base_variant in partition_of, f"{base_variant} not found in any manifest"
        assert f398_variant in partition_of, f"{f398_variant} not found in any manifest"
        assert partition_of[base_variant] == partition_of[f398_variant], (
            f"VID005 variants split across partitions: "
            f"{base_variant}={partition_of[base_variant]}, "
            f"{f398_variant}={partition_of[f398_variant]}"
        )


class TestAllExercisesProduceManifests:
    """test_all_exercises_produce_manifests: All 10 exercises get train/val/test manifest files."""

    def test_all_exercises(self, tmp_path):
        lm_dir = tmp_path / "landmarks"
        lm_dir.mkdir()
        splits_dir = tmp_path / "splits"
        splits_dir.mkdir()

        # Create minimal fake data for each exercise (need enough files for a 3-way split)
        for ex in EXERCISES:
            for i in range(6):
                label = "correct" if i < 3 else "incorrect"
                make_fake_csv(lm_dir, f"youtube_{ex}_{label}_VID{i:03d}.csv")

        # Run split for all exercises
        for ex in EXERCISES:
            split_exercise_videos(ex, lm_dir, splits_dir)

        # Verify manifest files were created for all exercises
        missing = []
        for ex in EXERCISES:
            for part in ("train", "val", "test"):
                manifest = splits_dir / f"{ex}_{part}.txt"
                if not manifest.exists():
                    missing.append(str(manifest))

        assert missing == [], f"Missing manifest files:\n" + "\n".join(missing)


class TestManifestIdempotent:
    """test_manifest_idempotent: Running twice without --force exits without overwriting."""

    def test_no_overwrite_without_force(self, fake_landmarks_dir, splits_dir):
        # First run
        split_exercise_videos("squat", fake_landmarks_dir, splits_dir)
        train_file = splits_dir / "squat_train.txt"
        assert train_file.exists()
        original_content = train_file.read_text()
        original_mtime = train_file.stat().st_mtime

        # Second run without force — should NOT overwrite
        result = split_exercise_videos("squat", fake_landmarks_dir, splits_dir, force=False)
        assert result == "skipped", f"Expected 'skipped' return value, got {result!r}"

        # File content unchanged
        new_content = train_file.read_text()
        assert new_content == original_content, "Manifest was overwritten without --force"

    def test_force_flag_overwrites(self, fake_landmarks_dir, splits_dir):
        # First run
        split_exercise_videos("squat", fake_landmarks_dir, splits_dir)
        train_file = splits_dir / "squat_train.txt"
        original_mtime = train_file.stat().st_mtime

        import time
        time.sleep(0.05)  # Ensure mtime would differ if file is rewritten

        # Second run WITH force — should overwrite
        result = split_exercise_videos("squat", fake_landmarks_dir, splits_dir, force=True)
        assert result != "skipped", f"Expected overwrite but got 'skipped'"


class TestClassRepresentation:
    """test_class_representation: Each partition has both correct and incorrect labels."""

    def test_both_classes_in_train(self, manifests):
        train, val, test = manifests
        has_correct = any("_correct_" in f for f in train)
        has_incorrect = any("_incorrect_" in f for f in train)
        # With 10 unique base videos (5 correct, 5 incorrect), train should have both
        assert has_correct, "Train manifest has no 'correct' samples"
        assert has_incorrect, "Train manifest has no 'incorrect' samples"
