"""
Retrain all exercises using Optuna-optimal hyperparameters.
Reads best params from reports/{exercise}_optuna_best.json,
trains with 500 epochs + full SWA/warm restarts pipeline.
3 parallel workers.
"""
import sys
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = r"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"
TRAIN_SCRIPT = str(PROJECT_ROOT / "scripts" / "train_bilstm.py")
REPORTS_DIR = PROJECT_ROOT / "reports"


def retrain_exercise(exercise: str) -> dict:
    """Retrain one exercise with its Optuna-optimal params."""
    params_file = REPORTS_DIR / f"{exercise}_optuna_best.json"
    if not params_file.exists():
        return {"exercise": exercise, "status": "skip", "elapsed": 0, "output_tail": "No optuna params"}

    with open(params_file) as f:
        data = json.load(f)
    p = data["best_params"]

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--use-landmarks",
        "--exercise", exercise,
        "--epochs", "500",
        "--batch-size", str(p["batch_size"]),
        "--lr", str(p["lr"]),
        "--hidden-dim", str(p["hidden_dim"]),
        "--num-layers", str(p["num_layers"]),
        "--dropout", str(p["dropout"]),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=7200,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr

        # Extract F1 from output
        f1 = "?"
        for line in output.split("\n"):
            if "F1:" in line and "Train" not in line and "val" not in line.lower():
                import re
                m = re.search(r"F1:\s+([0-9.]+)", line)
                if m:
                    f1 = m.group(1)

        return {
            "exercise": exercise,
            "status": "ok" if result.returncode == 0 else "error",
            "elapsed": round(elapsed, 1),
            "f1": f1,
            "params": p,
            "output_tail": output[-500:] if output else "",
        }
    except subprocess.TimeoutExpired:
        return {"exercise": exercise, "status": "timeout", "elapsed": 7200, "f1": "?", "params": p, "output_tail": ""}
    except Exception as e:
        return {"exercise": exercise, "status": "error", "elapsed": 0, "f1": "?", "params": p, "output_tail": str(e)}


def main():
    exercises = [
        "squat", "lunge", "deadlift", "bench_press", "overhead_press",
        "pullup", "pushup", "plank", "bicep_curl", "tricep_dip",
    ]
    workers = 3

    print(f"Retraining with Optuna-Optimal Configs")
    print(f"  Exercises: {len(exercises)}")
    print(f"  Workers:   {workers}")
    print(f"  Epochs:    500 (with early stopping)")
    print(f"  Features:  AMP, warm restarts, mixup, SWA")
    print(f"{'='*60}")

    for ex in exercises:
        pf = REPORTS_DIR / f"{ex}_optuna_best.json"
        if pf.exists():
            with open(pf) as f:
                d = json.load(f)
            p = d["best_params"]
            print(f"  {ex:<18} h={p['hidden_dim']}, L={p['num_layers']}, "
                  f"d={p['dropout']}, bs={p['batch_size']}, lr={p['lr']:.5f}")
    print(f"{'='*60}\n")

    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(retrain_exercise, ex): ex for ex in exercises}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            icon = "OK" if r["status"] == "ok" else "FAIL"
            print(f"  {icon} {r['exercise']:<18} F1={r.get('f1','?')} ({r['elapsed']:.0f}s)")
            if r["status"] != "ok":
                print(f"     {r['output_tail'][-200:]}")

    wall = time.time() - start
    print(f"\n{'='*60}")
    print(f"RETRAINING COMPLETE in {wall:.0f}s ({wall/60:.1f} min)")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"  Succeeded: {ok}/{len(exercises)}")


if __name__ == "__main__":
    main()
