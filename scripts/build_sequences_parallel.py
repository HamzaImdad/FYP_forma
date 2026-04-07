"""
Build sequences for ALL exercises in parallel using all CPU cores.

Launches 10 concurrent subprocesses (one per exercise) to max out
the i7-14700HX (20 cores / 28 threads). Each subprocess runs
build_sequences.py for a single exercise.

Usage:
    python scripts/build_sequences_parallel.py
"""

import sys
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES

PYTHON = r"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"
BUILD_SCRIPT = str(PROJECT_ROOT / "scripts" / "build_sequences.py")

# Per-exercise sequence lengths
EXERCISE_SEQ_LENS = {
    "deadlift": 45, "squat": 45,
    "bench_press": 30, "pushup": 30, "pullup": 30,
    "plank": 30, "lunge": 30, "bicep_curl": 30,
    "overhead_press": 30, "tricep_dip": 30,
}


def build_exercise(exercise: str) -> dict:
    """Build sequences for one exercise as a subprocess."""
    seq_len = EXERCISE_SEQ_LENS.get(exercise, 30)
    cmd = [
        PYTHON, BUILD_SCRIPT,
        "--use-landmarks",
        "--temporal",
        "--exercise", exercise,
        "--seq-len", str(seq_len),
        "--no-per-exercise-seqlen",  # we set it manually above
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=1800,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        return {
            "exercise": exercise,
            "seq_len": seq_len,
            "status": "ok" if result.returncode == 0 else "error",
            "elapsed": round(elapsed, 1),
            "output_tail": output[-500:] if output else "",
        }
    except subprocess.TimeoutExpired:
        return {"exercise": exercise, "seq_len": seq_len,
                "status": "timeout", "elapsed": 1800, "output_tail": ""}
    except Exception as e:
        return {"exercise": exercise, "seq_len": seq_len,
                "status": "error", "elapsed": time.time() - start,
                "output_tail": str(e)}


def main():
    exercises = list(EXERCISES)
    workers = len(exercises)  # 10 exercises = 10 parallel processes

    print(f"Parallel Sequence Building")
    print(f"  Exercises: {len(exercises)}")
    print(f"  Workers:   {workers} (all exercises simultaneously)")
    print(f"  CPU:       i7-14700HX (20 cores / 28 threads)")
    print(f"{'='*60}\n")

    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_exercise, ex): ex
            for ex in exercises
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            icon = "OK" if result["status"] == "ok" else "FAIL"
            print(f"  {icon} {result['exercise']:<18} seq_len={result['seq_len']} "
                  f"({result['elapsed']:.0f}s)")
            if result["status"] != "ok":
                print(f"     {result['output_tail'][-200:]}")

    wall = time.time() - start_time
    total_cpu = sum(r["elapsed"] for r in results)
    ok = sum(1 for r in results if r["status"] == "ok")

    print(f"\n{'='*60}")
    print(f"DONE: {ok}/{len(exercises)} succeeded")
    print(f"  Wall time: {wall:.0f}s ({wall/60:.1f} min)")
    print(f"  CPU time:  {total_cpu:.0f}s ({total_cpu/60:.1f} min)")
    print(f"  Speedup:   {total_cpu/max(wall,1):.1f}x")


if __name__ == "__main__":
    main()
