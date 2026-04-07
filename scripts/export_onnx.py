"""
Export all BiLSTM v2 models to ONNX format for faster inference via ONNX Runtime.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --exercise squat deadlift
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.bilstm_classifier import FormBiLSTM
from src.utils.constants import EXERCISES


def export_exercise(exercise: str, models_dir: Path) -> bool:
    """Export a single exercise's BiLSTM to ONNX. Returns True on success."""
    pt_path = models_dir / f"{exercise}_bilstm_v2.pt"
    onnx_path = models_dir / f"{exercise}_bilstm_v2.onnx"

    if not pt_path.exists():
        print(f"  [{exercise}] No checkpoint found at {pt_path}, skipping")
        return False

    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)

    input_dim = checkpoint.get("input_dim", 99)
    hidden_dim = checkpoint.get("hidden_dim", 128)
    num_layers = checkpoint.get("num_layers", 2)
    dropout = checkpoint.get("dropout", 0.3)
    use_conv = checkpoint.get("use_conv", True)
    seq_len = checkpoint.get("seq_len", 30)

    model = FormBiLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_conv=use_conv,
    )

    state_dict = checkpoint["model_state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()
               if not k.startswith("n_averaged")}  # Skip SWA artifact keys
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    dummy_input = torch.randn(1, seq_len, input_dim)

    # Verify PyTorch output
    with torch.no_grad():
        pt_output = model(dummy_input).item()

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 1: "seq_len"},
            "output": {0: "batch"},
        },
    )

    # Verify ONNX output matches PyTorch
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_output = sess.run(None, {"input": dummy_input.numpy()})[0].item()
        diff = abs(pt_output - onnx_output)
        if diff > 0.001:
            print(f"  [{exercise}] WARNING: output mismatch pt={pt_output:.4f} onnx={onnx_output:.4f} diff={diff:.6f}")
        else:
            print(f"  [{exercise}] OK — dim={input_dim} seq={seq_len} onnx={onnx_path.name} (diff={diff:.6f})")
        return True
    except ImportError:
        print(f"  [{exercise}] Exported to {onnx_path.name} (onnxruntime not installed, skipping verification)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Export BiLSTM models to ONNX")
    parser.add_argument("--exercise", nargs="+", help="Specific exercises to export")
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models" / "trained"
    exercises = args.exercise if args.exercise else EXERCISES

    print(f"Exporting {len(exercises)} exercises to ONNX...")
    success = 0
    for ex in exercises:
        if export_exercise(ex, models_dir):
            success += 1

    print(f"\nDone: {success}/{len(exercises)} models exported")


if __name__ == "__main__":
    main()
