#!/usr/bin/env python3
"""Rigorous ALPR detector latency benchmark.

Measures three distinct stages:
(a) Model inference only (GPU forward pass on pre-allocated tensor)
(b) Preprocess + inference + postprocess (from in-memory numpy array)
(c) Full pipeline including image decode (from disk file path)

Uses CUDA events and torch.cuda.synchronize() for microsecond-accurate GPU timing.
Executes >= 50 warm-up runs and >= 500 timed runs.
Reports mean, p50, p95, and p99 latencies in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def compute_statistics(latencies_ms: List[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "std_ms": round(float(np.std(arr)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "fps_mean": round(1000.0 / float(np.mean(arr)), 1),
    }


def benchmark_detector(
    weights_path: Path,
    sample_image_path: Path,
    warmup_runs: int = 50,
    timed_runs: int = 500,
    imgsz: int = 640,
    device: str = "cuda:0",
) -> Dict[str, dict]:
    print(f"Loading weights from {weights_path} onto {device}...")
    model = YOLO(str(weights_path))
    model.to(device)

    # Ensure CUDA is available
    if "cuda" in device and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but CUDA device was requested.")

    torch_model = model.model
    torch_model.eval()

    # Pre-allocate tensor for (a)
    dummy_tensor = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device=device)

    # Pre-load in-memory image for (b)
    img_bgr = cv2.imread(str(sample_image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Sample image not found or unreadable: {sample_image_path}")

    print(f"Running {warmup_runs} warm-up iterations across all stages...")
    # Warm-up (a)
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = torch_model(dummy_tensor)
        torch.cuda.synchronize()

    # Warm-up (b)
    for _ in range(warmup_runs):
        _ = model.predict(source=img_bgr, imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()

    # Warm-up (c)
    for _ in range(warmup_runs):
        _ = model.predict(source=str(sample_image_path), imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()

    print(f"Benchmarking (a) Model inference only ({timed_runs} runs)...")
    latencies_a = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(timed_runs):
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            _ = torch_model(dummy_tensor)
        end_event.record()
        torch.cuda.synchronize()
        latencies_a.append(start_event.elapsed_time(end_event))

    print(f"Benchmarking (b) Preprocess + inference + postprocess ({timed_runs} runs)...")
    latencies_b = []
    for _ in range(timed_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(source=img_bgr, imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_b.append((t1 - t0) * 1000.0)

    print(f"Benchmarking (c) Including image decode from disk ({timed_runs} runs)...")
    latencies_c = []
    for _ in range(timed_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(source=str(sample_image_path), imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_c.append((t1 - t0) * 1000.0)

    stats_a = compute_statistics(latencies_a)
    stats_b = compute_statistics(latencies_b)
    stats_c = compute_statistics(latencies_c)

    results = {
        "model_weights": str(weights_path),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "sample_image": str(sample_image_path),
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "imgsz": imgsz,
        "stage_a_inference_only": stats_a,
        "stage_b_preprocess_inference_postprocess": stats_b,
        "stage_c_including_disk_decode": stats_c,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Rigorous ALPR detector latency benchmark.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/runs/detect/ealpr_baseline_yolov8n/weights/best.pt"),
        help="Path to YOLO weights best.pt",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master\EALPR Vechicles dataset\Vehicles\0001.jpg"),
        help="Sample vehicle image for benchmarking.",
    )
    parser.add_argument("--warmup", type=int, default=50, help="Number of warm-up runs (default: 50)")
    parser.add_argument("--runs", type=int, default=500, help="Number of timed benchmark runs (default: 500)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (default: 640)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark/latency_report_500_runs.json"))
    args = parser.parse_args()

    results = benchmark_detector(
        weights_path=args.weights,
        sample_image_path=args.image,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
        imgsz=args.imgsz,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("DETECTOR LATENCY BENCHMARK RESULTS (NVIDIA RTX 4060 8GB, CUDA 12.4)")
    print("=" * 70)
    for name, key in [
        ("(a) Model Inference Only", "stage_a_inference_only"),
        ("(b) Preprocess + Inference + Postprocess", "stage_b_preprocess_inference_postprocess"),
        ("(c) Including Disk Image Decode", "stage_c_including_disk_decode"),
    ]:
        s = results[key]
        print(f"\n{name}:")
        print(f"  Mean:  {s['mean_ms']:6.2f} ms ({s['fps_mean']} FPS)")
        print(f"  p50:   {s['p50_ms']:6.2f} ms")
        print(f"  p95:   {s['p95_ms']:6.2f} ms")
        print(f"  p99:   {s['p99_ms']:6.2f} ms")
        print(f"  Std:   {s['std_ms']:6.2f} ms (Min: {s['min_ms']:.2f} ms, Max: {s['max_ms']:.2f} ms)")
    print("\n" + "=" * 70)
    print(f"Full JSON report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
