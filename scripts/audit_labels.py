"""
Audit label quality by finding samples where model predictions disagree
with ground truth labels. Helps identify mislabelled training data.

Usage:
    python scripts/audit_labels.py --exercise tricep_dip
    python scripts/audit_labels.py  (all weak exercises)
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES

SEQUENCES_DIR = PROJECT_ROOT / "data" / "processed" / "sequences"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default: audit all exercises
WEAK_EXERCISES = ["squat", "lunge", "deadlift", "bench_press", "overhead_press",
                  "pullup", "pushup", "plank", "bicep_curl", "tricep_dip"]


def load_all_sequences(exercise: str) -> tuple:
    """Load and concatenate train + val + test sequences.

    Returns: (X_all, y_all, video_ids_all, feature_names)
             All None if any partition is missing.
    """
    X_parts, y_parts, vid_parts = [], [], []
    feature_names = None

    for partition in ("train", "val", "test"):
        path = SEQUENCES_DIR / f"{exercise}_{partition}_sequences.npz"
        if not path.exists():
            print(f"  [WARNING] Missing: {path.name}")
            return None, None, None, None
        data = np.load(path, allow_pickle=True)
        X_parts.append(data["X"])
        y_parts.append(data["y"])
        if "video_ids" in data:
            vid_parts.append(data["video_ids"])
        else:
            vid_parts.append(np.array([f"{partition}_{i}" for i in range(len(data["y"]))]))
        if feature_names is None:
            feature_names = data["feature_names"]

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    video_ids_all = np.concatenate(vid_parts, axis=0)

    return X_all, y_all, video_ids_all, feature_names


def load_model(exercise: str, device: torch.device) -> tuple:
    """Load a trained BiLSTM v2 model.

    Returns: (model, threshold) or (None, None) if not found.
    """
    model_path = MODELS_DIR / f"{exercise}_bilstm_v2.pt"
    if not model_path.exists():
        model_path = MODELS_DIR / f"{exercise}_bilstm.pt"
    if not model_path.exists():
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_dim = checkpoint.get("input_dim", 99)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_layers", 2)
    n_heads = checkpoint.get("n_heads", 1)
    use_conv = checkpoint.get("use_conv", False)

    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
        use_conv=use_conv,
    )

    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)

    model.to(device)
    model.eval()

    threshold = checkpoint.get("best_threshold",
                               checkpoint.get("val_best_thresh_info", 0.5))

    return model, threshold


def audit_exercise(exercise: str, device: torch.device, confidence_threshold: float = 0.8):
    """Run label audit for a single exercise.

    Finds sequences where the model is confident (> threshold) but disagrees
    with the ground truth label. Groups flagged sequences by video_id.

    Returns: audit result dict, or None on failure.
    """
    print(f"\n{'='*60}")
    print(f"Label Audit: {exercise}")
    print(f"{'='*60}")

    # Load model
    model, model_threshold = load_model(exercise, device)
    if model is None:
        print(f"  [SKIP] No trained model found for {exercise}")
        return None

    # Load all sequences
    X_all, y_all, video_ids, feature_names = load_all_sequences(exercise)
    if X_all is None:
        print(f"  [SKIP] Sequences not found for {exercise}")
        return None

    # Check feature dim compatibility
    input_dim = X_all.shape[2]
    model_input_dim = list(model.parameters())[0].shape[-1] if hasattr(model, 'parameters') else input_dim
    # Get actual input dim from checkpoint
    model_path = MODELS_DIR / f"{exercise}_bilstm_v2.pt"
    ckpt_check = torch.load(model_path, map_location='cpu', weights_only=False)
    model_expected_dim = ckpt_check.get('input_dim', 99)
    if input_dim != model_expected_dim:
        print(f"  [SKIP] Feature dim mismatch: sequences={input_dim}, model={model_expected_dim}")
        return None

    print(f"  Total sequences: {len(y_all)}")
    print(f"  Correct: {int((y_all == 1).sum())}, Incorrect: {int((y_all == 0).sum())}")
    print(f"  Model threshold: {model_threshold:.3f}")
    print(f"  Confidence threshold for flagging: {confidence_threshold:.2f}")

    # Run inference on all sequences
    batch_size = 128
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_all), batch_size):
            batch = torch.tensor(X_all[i:i+batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)

    # Flag disagreements: model confident but disagrees with label
    flagged = []
    for idx in range(len(y_all)):
        prob = float(all_probs[idx])
        label = int(y_all[idx])
        pred = 1 if prob >= model_threshold else 0
        model_confidence = prob if pred == 1 else (1.0 - prob)

        # Model is confident AND disagrees with ground truth
        if model_confidence >= confidence_threshold and pred != label:
            flagged.append({
                "index": idx,
                "video_id": str(video_ids[idx]),
                "ground_truth": label,
                "predicted": pred,
                "probability": round(prob, 4),
                "confidence": round(model_confidence, 4),
            })

    # Group by video_id
    video_flags = defaultdict(list)
    for item in flagged:
        video_flags[item["video_id"]].append(item)

    # Sort videos by number of flagged sequences (most suspicious first)
    video_summary = []
    for vid, items in sorted(video_flags.items(), key=lambda x: -len(x[1])):
        gt_labels = [it["ground_truth"] for it in items]
        avg_conf = np.mean([it["confidence"] for it in items])
        video_summary.append({
            "video_id": vid,
            "flagged_count": len(items),
            "avg_confidence": round(float(avg_conf), 4),
            "ground_truth_label": gt_labels[0],  # same video = same label
        })

    flagged_pct = len(flagged) / len(y_all) * 100 if len(y_all) > 0 else 0.0

    print(f"\n  Results:")
    print(f"    Flagged sequences: {len(flagged)} / {len(y_all)} ({flagged_pct:.1f}%)")
    print(f"    Flagged videos:    {len(video_summary)}")

    if video_summary:
        print(f"\n  Top suspicious videos (model confident but disagrees with label):")
        for vs in video_summary[:15]:
            gt_str = "correct" if vs["ground_truth_label"] == 1 else "incorrect"
            vid_id = vs['video_id'].encode('ascii', 'replace').decode('ascii')
            print(f"    {vid_id}: {vs['flagged_count']} flagged seqs, "
                  f"avg_conf={vs['avg_confidence']:.3f}, labeled={gt_str}")

    # Build audit result
    result = {
        "exercise": exercise,
        "total_sequences": len(y_all),
        "flagged_count": len(flagged),
        "flagged_percentage": round(flagged_pct, 2),
        "confidence_threshold": confidence_threshold,
        "model_threshold": float(model_threshold),
        "flagged_videos": video_summary,
        "flagged_details": flagged,
    }

    # Save report
    out_path = REPORTS_DIR / f"{exercise}_label_audit.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Audit label quality for BiLSTM v2 models")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Exercise to audit (default: all weak exercises)")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Confidence threshold for flagging disagreements (default: 0.8)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.exercise:
        exercises = [args.exercise]
    else:
        exercises = WEAK_EXERCISES
        print(f"Auditing weak exercises (F1 < 0.60): {', '.join(exercises)}")

    all_results = {}
    for exercise in exercises:
        result = audit_exercise(exercise, device, confidence_threshold=args.threshold)
        if result:
            all_results[exercise] = result

    # Summary table
    if all_results:
        print(f"\n{'='*60}")
        print(f"{'Exercise':<20} {'Total':>8} {'Flagged':>8} {'Pct':>8} {'Videos':>8}")
        print(f"{'-'*60}")
        for ex, res in all_results.items():
            print(f"{ex:<20} {res['total_sequences']:>8} {res['flagged_count']:>8} "
                  f"{res['flagged_percentage']:>7.1f}% {len(res['flagged_videos']):>8}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
