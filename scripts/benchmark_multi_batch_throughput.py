#!/usr/bin/env python3
"""Rigorous Multi-Batch Latency and Throughput Benchmark (Mission 5 Section 5).
Evaluates YOLOv8n detector across FP32 vs FP16 and batch sizes [1, 4, 8, 16] on RTX 4060.
Measures forward-pass inference latency and throughput using CUDA events and synchronization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_benchmark_configuration(
    model_torch: torch.nn.Module,
    batch_size: int,
    imgsz: int,
    precision: str,  # 'fp32' or 'fp16'
    device: torch.device,
    warmup_runs: int = 50,
    timed_runs: int = 200,
) -> dict:
    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    # Put model in desired precision
    if precision == "fp16":
        model_torch.half()
    else:
        model_torch.float()

    model_torch.eval()

    # Pre-allocate dummy input tensor on GPU
    dummy_input = torch.zeros((batch_size, 3, imgsz, imgsz), dtype=torch_dtype, device=device)

    # Reset peak memory
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model_torch(dummy_input)
        torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_ms: List[float] = []

    for _ in range(timed_runs):
        torch.cuda.synchronize(device)
        start_event.record()

        with torch.no_grad():
            _ = model_torch(dummy_input)

        end_event.record()
        torch.cuda.synchronize(device)

        batch_latency = start_event.elapsed_time(end_event)  # milliseconds for batch
        latencies_ms.append(batch_latency)

    lat_arr = np.array(latencies_ms)
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    mean_batch_lat = float(np.mean(lat_arr))
    p50_batch_lat = float(np.percentile(lat_arr, 50))
    p95_batch_lat = float(np.percentile(lat_arr, 95))
    p99_batch_lat = float(np.percentile(lat_arr, 99))

    # Per-image latency
    mean_img_lat = mean_batch_lat / batch_size
    p50_img_lat = p50_batch_lat / batch_size
    p95_img_lat = p95_batch_lat / batch_size
    p99_img_lat = p99_batch_lat / batch_size

    # Throughput in images per second
    throughput_fps = (batch_size * 1000.0) / mean_batch_lat

    return {
        "precision": precision.upper(),
        "batch_size": batch_size,
        "image_size": f"{imgsz}x{imgsz}",
        "warmup_runs": warmup_runs,
        "timed_runs": timed_runs,
        "timing_scope": "Model forward inference only (pre-allocated GPU tensor, excluded disk decode/transfer)",
        "batch_latency_ms": {
            "mean": round(mean_batch_lat, 3),
            "p50": round(p50_batch_lat, 3),
            "p95": round(p95_batch_lat, 3),
            "p99": round(p99_batch_lat, 3),
        },
        "per_image_latency_ms": {
            "mean": round(mean_img_lat, 3),
            "p50": round(p50_img_lat, 3),
            "p95": round(p95_img_lat, 3),
            "p99": round(p99_img_lat, 3),
        },
        "throughput_images_per_sec": round(throughput_fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-batch FP32 vs FP16 latency and throughput benchmark.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark_reports"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. GPU is required for throughput benchmarking.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Benchmarking on GPU: {gpu_name}")
    print(f"Loading checkpoint: {args.weights}")

    yolo = YOLO(str(args.weights))
    torch_model = yolo.model.to(device)

    batch_sizes = [1, 4, 8, 16]
    precisions = ["fp32", "fp16"]

    results = []

    print("\n" + "=" * 90)
    print(f"{'Precision':<10} | {'Batch':<6} | {'Batch Lat (ms)':<15} | {'Per-Img Lat (ms)':<16} | {'Throughput (FPS)':<18} | {'Peak VRAM (MB)':<12}")
    print("=" * 90)

    for prec in precisions:
        for b in batch_sizes:
            cfg_res = run_benchmark_configuration(
                model_torch=torch_model,
                batch_size=b,
                imgsz=args.imgsz,
                precision=prec,
                device=device,
                warmup_runs=args.warmup,
                timed_runs=args.runs,
            )
            results.append(cfg_res)

            mean_b = cfg_res["batch_latency_ms"]["mean"]
            mean_i = cfg_res["per_image_latency_ms"]["mean"]
            fps = cfg_res["throughput_images_per_sec"]
            vram = cfg_res["peak_vram_mb"]

            print(f"{prec.upper():<10} | {b:<6} | {mean_b:<15.3f} | {mean_i:<16.3f} | {fps:<18.1f} | {vram:<12.1f}")

    print("=" * 90)

    # Save machine-readable JSON
    report = {
        "hardware": {
            "gpu": gpu_name,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        },
        "model": {
            "name": "YOLOv8n Plate Detector",
            "weights": str(args.weights),
            "input_resolution": f"{args.imgsz}x{args.imgsz}",
        },
        "benchmark_results": results,
    }

    json_path = args.output_dir / "multi_batch_throughput_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {json_path}")


if __name__ == "__main__":
    main()
