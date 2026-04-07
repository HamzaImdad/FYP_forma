"""
Parallel Optuna search — runs 3 exercises simultaneously.
30 trials, 50 max epochs, MedianPruner.
"""
import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
PYTHON = r"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"
OPTUNA_SCRIPT = str(PROJECT_ROOT / "scripts" / "optuna_search.py")


def search_exercise(exercise: str) -> dict:
    cmd = [
        PYTHON, OPTUNA_SCRIPT,
        "--exercise", exercise,
        "--n-trials", "30",
        "--max-epochs", "50",
    ]
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=3600,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        # Extract best F1 from output
        best_f1 = "?"
        for line in output.split("\n"):
            if "best_val_f1" in line:
                import re
                m = re.search(r"[0-9]\.[0-9]+", line)
                if m:
                    best_f1 = m.group()
        return {
            "exercise": exercise,
            "status": "ok" if result.returncode == 0 else "error",
            "elapsed": round(elapsed, 1),
            "best_f1": best_f1,
            "output_tail": output[-300:] if output else "",
        }
    except subprocess.TimeoutExpired:
        return {"exercise": exercise, "status": "timeout", "elapsed": 3600, "best_f1": "?", "output_tail": ""}
    except Exception as e:
        return {"exercise": exercise, "status": "error", "elapsed": 0, "best_f1": "?", "output_tail": str(e)}


def main():
    exercises = sys.argv[1:] if len(sys.argv) > 1 else ["pullup", "pushup", "plank", "bicep_curl", "tricep_dip"]
    workers = min(3, len(exercises))

    print(f"Parallel Optuna Search")
    print(f"  Exercises: {exercises}")
    print(f"  Workers:   {workers}")
    print(f"  Trials:    30 per exercise")
    print(f"  Max epochs: 50")
    print(f"{'='*50}")

    start = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(search_exercise, ex): ex for ex in exercises}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            icon = "OK" if r["status"] == "ok" else "FAIL"
            print(f"  {icon} {r['exercise']:<18} F1={r['best_f1']} ({r['elapsed']:.0f}s)")
            if r["status"] != "ok":
                print(f"     {r['output_tail'][-200:]}")

    wall = time.time() - start
    print(f"\n{'='*50}")
    print(f"Done in {wall:.0f}s ({wall/60:.1f} min)")


if __name__ == "__main__":
    main()
