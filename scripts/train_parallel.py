"""
Train BiLSTM v2/v3 models for multiple exercises in parallel.

Launches concurrent training processes to maximize GPU + CPU utilization.
Each process trains one exercise at a time on a shared GPU.
With AMP (FP16), each process uses ~1-2GB VRAM, allowing 4-5 concurrent on RTX 4070 8GB.

Hardware utilization:
  - GPU: 4-5 concurrent training processes sharing RTX 4070 via AMP
  - Shared GPU memory: overflow to system RAM when VRAM is full
  - CPU: Each subprocess uses 12 DataLoader workers (28 threads available total)
  - RAM: pin_memory=True, prefetch_factor=4 for fast CPU->GPU transfer

Usage:
    python scripts/train_parallel.py --workers 5 --epochs 500
    python scripts/train_parallel.py --workers 5 --epochs 500 --exercises squat lunge deadlift
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.constants import EXERCISES

PYTHON = r"C:/Users/Hamza/AppData/Local/Programs/Python/Python310/python.exe"
TRAIN_SCRIPT = str(PROJECT_ROOT / "scripts" / "train_bilstm.py")


def train_exercise(exercise: str, epochs: int, batch_size: int) -> dict:
    """Train a single exercise as a subprocess."""
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--use-landmarks",
        "--exercise", exercise,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    # Set environment to limit GPU memory per process for sharing
    env = os.environ.copy()
    env["CUDA_MPS_PIPE_DIRECTORY"] = ""  # Let CUDA manage memory sharing

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=7200,  # 2 hours max per exercise (500 epochs)
            env=env,
        )
        elapsed = time.time() - start

        output = result.stdout + result.stderr
        return {
            "exercise": exercise,
            "status": "ok" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "elapsed": round(elapsed, 1),
            "output_tail": output[-1000:] if output else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "exercise": exercise,
            "status": "timeout",
            "elapsed": 7200,
            "output_tail": "Process timed out after 2 hours",
        }
    except Exception as e:
        return {
            "exercise": exercise,
            "status": "error",
            "elapsed": time.time() - start,
            "output_tail": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM v3 models in parallel")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel training processes (default: 5, "
                             "fits 5 AMP processes in 8GB VRAM)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Training epochs per exercise (default: 500)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size per exercise (default: 128, AMP enables this)")
    parser.add_argument("--exercises", nargs="+", default=None,
                        help="Specific exercises to train (default: all 10)")
    args = parser.parse_args()

    exercises = args.exercises or list(EXERCISES)

    print(f"Parallel BiLSTM v3 Training")
    print(f"  Exercises:   {len(exercises)}")
    print(f"  Workers:     {args.workers}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  GPU:         RTX 4070 (8GB VRAM + shared)")
    print(f"  CPU:         i7-14700HX (20c/28t)")
    print(f"  AMP:         enabled (FP16)")
    print(f"  DataLoader:  12 workers/process, pin_memory, prefetch_factor=4")
    print(f"{'='*60}\n")

    completed = 0
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(train_exercise, ex, args.epochs, args.batch_size): ex
            for ex in exercises
        }

        for future in as_completed(futures):
            result = future.result()
            completed += 1
            results.append(result)

            status_icon = "OK" if result["status"] == "ok" else "FAIL"
            wall_elapsed = time.time() - start_time
            print(f"[{completed}/{len(exercises)}] {status_icon} {result['exercise']} "
                  f"({result['elapsed']:.0f}s, wall: {wall_elapsed:.0f}s)")

            if result["status"] != "ok":
                print(f"  Last output: {result['output_tail'][-300:]}")

    wall_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] != "ok")
    total_cpu_time = sum(r["elapsed"] for r in results)
    print(f"  Succeeded:  {ok}/{len(exercises)}")
    if fail:
        print(f"  Failed:     {fail}")
        for r in results:
            if r["status"] != "ok":
                print(f"    - {r['exercise']}: {r['status']}")
    print(f"  Wall time:  {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print(f"  Total CPU:  {total_cpu_time:.0f}s ({total_cpu_time/60:.1f} min)")
    print(f"  Speedup:    {total_cpu_time/max(wall_time, 1):.1f}x")


if __name__ == "__main__":
    main()
