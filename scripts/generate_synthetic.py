"""
Generate synthetic sequences via interpolation between existing samples.
Targets data-poor exercises to improve decision boundary learning.

Usage:
    python scripts/generate_synthetic.py --exercise tricep_dip --n-synthetic 500
    python scripts/generate_synthetic.py  (all data-poor exercises)
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"

# Default threshold: exercises with fewer than this many train sequences
# are considered data-poor and get synthetic augmentation
DATA_POOR_THRESHOLD = 5000


def load_train_sequences(exercise: str) -> tuple:
    """Load train partition sequences for an exercise.

    Returns: (X, y, video_ids, feature_names) or (None, None, None, None)
    """
    path = SEQUENCES_DIR / f"{exercise}_train_sequences.npz"
    if not path.exists():
        print(f"  [WARNING] Missing: {path.name}")
        return None, None, None, None

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"] if "video_ids" in data else np.array([f"train_{i}" for i in range(len(y))])
    feature_names = data["feature_names"]

    return X, y, video_ids, feature_names


def generate_synthetic_sequences(
    X_class: np.ndarray,
    n_synthetic: int,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generate synthetic sequences via pairwise interpolation (SMOTE-like).

    For each synthetic sample:
      1. Randomly select two sequences from the same class
      2. Draw lambda ~ Beta(0.5, 0.5) for U-shaped distribution
         (biases toward one parent or the other, avoids bland averages)
      3. Interpolate: x_new = lambda * x1 + (1 - lambda) * x2

    Args:
        X_class: Array of shape (n_samples, seq_len, n_features) for one class
        n_synthetic: Number of synthetic sequences to generate
        rng: Numpy random generator

    Returns:
        Synthetic sequences of shape (n_synthetic, seq_len, n_features)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(X_class)
    if n_samples < 2:
        print(f"    [WARNING] Need at least 2 samples to interpolate, got {n_samples}")
        return np.empty((0, *X_class.shape[1:]), dtype=X_class.dtype)

    seq_len, n_features = X_class.shape[1], X_class.shape[2]
    synthetic = np.empty((n_synthetic, seq_len, n_features), dtype=np.float32)

    # Draw all random indices and lambdas upfront for efficiency
    idx1 = rng.integers(0, n_samples, size=n_synthetic)
    idx2 = rng.integers(0, n_samples, size=n_synthetic)

    # Ensure idx1 != idx2 (resample collisions)
    collisions = idx1 == idx2
    while collisions.any():
        idx2[collisions] = rng.integers(0, n_samples, size=int(collisions.sum()))
        collisions = idx1 == idx2

    # Beta(0.5, 0.5) gives U-shaped distribution — samples near parents
    lambdas = rng.beta(0.5, 0.5, size=n_synthetic).astype(np.float32)

    # Vectorized interpolation
    for i in range(n_synthetic):
        lam = lambdas[i]
        synthetic[i] = lam * X_class[idx1[i]] + (1.0 - lam) * X_class[idx2[i]]

    return synthetic


def generate_for_exercise(
    exercise: str,
    n_synthetic: int = 1000,
    output_mode: str = "separate",
    rng: np.random.Generator = None,
) -> dict:
    """Generate synthetic sequences for one exercise.

    Args:
        exercise: Exercise name
        n_synthetic: Number of synthetic sequences to generate per class
        output_mode: "separate" saves as new file, "append" appends to train file
        rng: Numpy random generator

    Returns: Summary dict or None on failure
    """
    print(f"\n{'='*60}")
    print(f"Synthetic Generation: {exercise}")
    print(f"{'='*60}")

    X_train, y_train, video_ids, feature_names = load_train_sequences(exercise)
    if X_train is None:
        print(f"  [SKIP] No train sequences found")
        return None

    n_correct = int((y_train == 1).sum())
    n_incorrect = int((y_train == 0).sum())
    print(f"  Original train: {len(y_train)} (correct: {n_correct}, incorrect: {n_incorrect})")
    print(f"  Shape: {X_train.shape}")
    print(f"  Generating {n_synthetic} synthetic per class...")

    # Split by class
    X_correct = X_train[y_train == 1]
    X_incorrect = X_train[y_train == 0]

    # Generate synthetic for each class
    syn_correct = generate_synthetic_sequences(X_correct, n_synthetic, rng)
    syn_incorrect = generate_synthetic_sequences(X_incorrect, n_synthetic, rng)

    n_syn_correct = len(syn_correct)
    n_syn_incorrect = len(syn_incorrect)
    print(f"  Generated: {n_syn_correct} correct + {n_syn_incorrect} incorrect")

    if n_syn_correct == 0 and n_syn_incorrect == 0:
        print(f"  [SKIP] No synthetic sequences generated")
        return None

    # Combine synthetic sequences
    X_synthetic = np.concatenate(
        [s for s in [syn_correct, syn_incorrect] if len(s) > 0], axis=0
    )
    y_synthetic = np.concatenate([
        np.ones(n_syn_correct, dtype=np.float64),
        np.zeros(n_syn_incorrect, dtype=np.float64),
    ])
    vid_synthetic = np.array([f"synthetic_{i}" for i in range(len(y_synthetic))])

    if output_mode == "append":
        # Append to existing train data
        X_out = np.concatenate([X_train, X_synthetic], axis=0)
        y_out = np.concatenate([y_train, y_synthetic], axis=0)
        vid_out = np.concatenate([video_ids, vid_synthetic], axis=0)
        out_path = SEQUENCES_DIR / f"{exercise}_train_sequences.npz"
        action = "Appended to"
    else:
        # Save as separate file
        X_out = X_synthetic
        y_out = y_synthetic
        vid_out = vid_synthetic
        out_path = SEQUENCES_DIR / f"{exercise}_train_augmented_sequences.npz"
        action = "Saved to"

    np.savez_compressed(
        out_path,
        X=X_out,
        y=y_out,
        video_ids=vid_out,
        feature_names=feature_names,
    )

    total_correct = int((y_out == 1).sum())
    total_incorrect = int((y_out == 0).sum())
    print(f"  {action}: {out_path.name}")
    print(f"  Final: {len(y_out)} (correct: {total_correct}, incorrect: {total_incorrect})")

    return {
        "exercise": exercise,
        "original_train": len(y_train),
        "original_correct": n_correct,
        "original_incorrect": n_incorrect,
        "synthetic_correct": n_syn_correct,
        "synthetic_incorrect": n_syn_incorrect,
        "total_output": len(y_out),
        "output_file": out_path.name,
        "output_mode": output_mode,
    }


def find_data_poor_exercises(threshold: int = DATA_POOR_THRESHOLD) -> list:
    """Find exercises with fewer than `threshold` train sequences."""
    data_poor = []
    for exercise in EXERCISES:
        path = SEQUENCES_DIR / f"{exercise}_train_sequences.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        n = len(data["y"])
        if n < threshold:
            data_poor.append((exercise, n))
            print(f"  {exercise}: {n} train sequences (< {threshold})")

    return [ex for ex, _ in sorted(data_poor, key=lambda x: x[1])]


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sequences via interpolation (SMOTE-like)"
    )
    parser.add_argument("--exercise", type=str, default=None,
                        help="Exercise name (default: all data-poor exercises)")
    parser.add_argument("--n-synthetic", type=int, default=1000,
                        help="Number of synthetic sequences per class (default: 1000)")
    parser.add_argument("--output-mode", type=str, choices=["append", "separate"],
                        default="separate",
                        help="Output mode: 'separate' (new file) or 'append' (to train file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.exercise:
        exercises = [args.exercise]
    else:
        print(f"Finding data-poor exercises (< {DATA_POOR_THRESHOLD} train sequences)...")
        exercises = find_data_poor_exercises()
        if not exercises:
            print("No data-poor exercises found. Use --exercise to target a specific one.")
            return

    all_results = {}
    for exercise in exercises:
        result = generate_for_exercise(
            exercise,
            n_synthetic=args.n_synthetic,
            output_mode=args.output_mode,
            rng=rng,
        )
        if result:
            all_results[exercise] = result

    # Summary table
    if all_results:
        print(f"\n{'='*70}")
        print(f"{'Exercise':<20} {'Original':>10} {'Synthetic':>10} {'Total':>10} {'File'}")
        print(f"{'-'*70}")
        for ex, res in all_results.items():
            n_syn = res['synthetic_correct'] + res['synthetic_incorrect']
            print(f"{ex:<20} {res['original_train']:>10} {n_syn:>10} "
                  f"{res['total_output']:>10} {res['output_file']}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
