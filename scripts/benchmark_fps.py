"""
FPS benchmark for ExerVision pipeline.

Measures real-time performance of each classifier type by processing
synthetic frames through the full pipeline. Reports average FPS,
per-stage timings, and p95 latency.

Usage:
    python scripts/benchmark_fps.py
    python scripts/benchmark_fps.py --frames 300 --exercise squat
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.realtime import ExerVisionPipeline
from src.utils.constants import EXERCISES

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_frame(width=640, height=480):
    """Generate a random frame to simulate camera input."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


def benchmark_classifier(classifier_type, exercise, n_frames, warmup=10):
    """Benchmark a specific classifier type.

    Returns dict with timing statistics.
    """
    print(f"\n  Benchmarking {classifier_type} on {exercise} ({n_frames} frames)...")

    try:
        pipeline = ExerVisionPipeline(
            exercise=exercise,
            classifier_type=classifier_type,
            smooth=True,
        )
    except Exception as e:
        print(f"    Failed to create pipeline: {e}")
        return None

    frame = generate_synthetic_frame()
    frame_times = []

    # Warmup phase (not measured)
    for i in range(warmup):
        ts = int(i * (1000 / 30))
        try:
            pipeline.process_frame(frame.copy(), ts)
        except Exception:
            pass

    # Benchmark phase
    for i in range(n_frames):
        ts = int((warmup + i) * (1000 / 30))
        start = time.perf_counter()
        try:
            pipeline.process_frame(frame.copy(), ts)
        except Exception:
            pass
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)

    pipeline.close()

    times_ms = np.array(frame_times) * 1000
    fps_values = 1000.0 / times_ms

    results = {
        "classifier": classifier_type,
        "exercise": exercise,
        "n_frames": n_frames,
        "avg_fps": float(np.mean(fps_values)),
        "median_fps": float(np.median(fps_values)),
        "min_fps": float(np.min(fps_values)),
        "max_fps": float(np.max(fps_values)),
        "avg_latency_ms": float(np.mean(times_ms)),
        "median_latency_ms": float(np.median(times_ms)),
        "p95_latency_ms": float(np.percentile(times_ms, 95)),
        "p99_latency_ms": float(np.percentile(times_ms, 99)),
    }

    meets_target = results["avg_fps"] >= 15
    status = "PASS" if meets_target else "FAIL"

    print(f"    Avg FPS:       {results['avg_fps']:.1f}")
    print(f"    Median FPS:    {results['median_fps']:.1f}")
    print(f"    Avg latency:   {results['avg_latency_ms']:.1f} ms")
    print(f"    P95 latency:   {results['p95_latency_ms']:.1f} ms")
    print(f"    Target (>=15 FPS): [{status}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark ExerVision FPS")
    parser.add_argument("--frames", type=int, default=300,
                        help="Number of frames to process (default: 300)")
    parser.add_argument("--exercise", type=str, default=None,
                        help="Benchmark a specific exercise only")
    parser.add_argument("--classifiers", type=str, default="rule_based,ml,bilstm",
                        help="Comma-separated classifier types")
    args = parser.parse_args()

    classifier_types = [c.strip() for c in args.classifiers.split(",")]
    exercises = [args.exercise] if args.exercise else ["squat", "pushup", "bicep_curl"]

    print("ExerVision FPS Benchmark")
    print("=" * 60)
    print(f"Frames per test: {args.frames}")
    print(f"Classifiers: {', '.join(classifier_types)}")
    print(f"Exercises: {', '.join(exercises)}")

    all_results = []

    for exercise in exercises:
        print(f"\n{'='*60}")
        print(f"Exercise: {exercise}")

        for clf_type in classifier_types:
            result = benchmark_classifier(clf_type, exercise, args.frames)
            if result:
                all_results.append(result)

    # Summary table
    if all_results:
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"{'Classifier':<15} {'Exercise':<15} {'Avg FPS':>10} {'P95 Lat(ms)':>12} {'Status':>8}")
        print("-" * 62)

        for r in all_results:
            status = "PASS" if r["avg_fps"] >= 15 else "FAIL"
            print(f"{r['classifier']:<15} {r['exercise']:<15} "
                  f"{r['avg_fps']:>10.1f} {r['p95_latency_ms']:>12.1f} {status:>8}")

        # Save results
        results_path = REPORTS_DIR / "fps_benchmark.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        # Check overall pass/fail
        all_pass = all(r["avg_fps"] >= 15 for r in all_results)
        if all_pass:
            print("\nAll classifiers meet the 15 FPS requirement.")
        else:
            failing = [f"{r['classifier']}/{r['exercise']}" for r in all_results if r["avg_fps"] < 15]
            print(f"\nBelow 15 FPS: {', '.join(failing)}")


if __name__ == "__main__":
    main()
