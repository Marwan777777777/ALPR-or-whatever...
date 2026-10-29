#!/usr/bin/env python3
"""CPU Reality Check Benchmark (Mission 007 Step 5).

Runs full ALPR pipeline (detector + character model) on device='cpu'
over 50 gate triage images (batch 1).
Reports mean and p95 latency in milliseconds.
Labels output: 'dev laptop CPU, not the gate server'.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
import sys
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.pipeline.alpr_pipeline import EndToEndALPRPipeline
from scripts.run_gate_triage import read_image_utf8


def benchmark_cpu(gate_dir: Path, num_images: int = 50, warmup_runs: int = 5) -> dict:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_imgs = sorted([f for f in gate_dir.rglob("*") if f.suffix.lower() in valid_exts])
    if len(all_imgs) < num_images:
        num_images = len(all_imgs)

    sample_paths = all_imgs[:num_images]
    print(f"Loading {num_images} gate images for CPU benchmark...")
    loaded_images = []
    for p in sample_paths:
        im = read_image_utf8(p)
        if im is not None:
            loaded_images.append(im)
        if len(loaded_images) >= num_images:
            break

    print(f"Loaded {len(loaded_images)} valid images into RAM.")

    print("Initializing EndToEndALPRPipeline with device='cpu'...")
    pipeline = EndToEndALPRPipeline(device="cpu", agnostic_nms=False)

    # Warmup
    print(f"Running {warmup_runs} warmup runs on CPU...")
    for i in range(min(warmup_runs, len(loaded_images))):
        _ = pipeline.read_plate_image(loaded_images[i], source_id=f"warmup_{i}")

    # Latency benchmark
    print(f"Benchmarking {len(loaded_images)} images on dev laptop CPU (batch 1)...")
    latencies_ms: List[float] = []

    for idx, im in enumerate(loaded_images, 1):
        t0 = time.perf_counter()
        _ = pipeline.read_plate_image(im, source_id=f"cpu_bench_{idx}")
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    p50_ms = float(np.percentile(latencies_ms, 50))
    p95_ms = float(np.percentile(latencies_ms, 95))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    has_onnx = (importlib.util.find_spec("onnxruntime") is not None)

    result = {
        "hardware_label": "dev laptop CPU, not the gate server",
        "device": "cpu",
        "batch_size": 1,
        "num_images_benchmarked": len(latencies_ms),
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": round(p50_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "min_latency_ms": round(min_ms, 2),
        "max_latency_ms": round(max_ms, 2),
        "throughput_fps": round(fps, 2),
        "onnxruntime_status": "NOT_INSTALLED (not installed per strict constraints, ONNX export benchmark skipped)",
    }

    out_path = Path("artifacts/triage/cpu_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved CPU benchmark results to: {out_path}")

    return result


if __name__ == "__main__":
    gate_path = Path(r"D:\TruckEntry")
    res = benchmark_cpu(gate_path, num_images=50)
    print("\n" + "=" * 60)
    print(f"CPU REALITY CHECK: {res['hardware_label']}")
    print("=" * 60)
    print(f"Mean Latency: {res['mean_latency_ms']} ms ({res['throughput_fps']} FPS)")
    print(f"p95 Latency:  {res['p95_latency_ms']} ms")
    print(f"ONNX Status:  {res['onnxruntime_status']}")
