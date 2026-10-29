#!/usr/bin/env python3
"""Trains the baseline Egyptian License Plate Detector on the EALPR dataset.

Enforces GPU acceleration (fails explicitly if CUDA is unavailable), strict reproducibility,
and saves artifacts to git-ignored directories.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.logging import setup_logger


def verify_gpu_cuda() -> dict:
    """Verifies that PyTorch detects the NVIDIA GPU and runs genuine CUDA operations."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. Refusing to silently fallback to CPU per Mission 003 contract."
        )

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_capability = torch.cuda.get_device_capability(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = round(vram_bytes / (1024 ** 3), 2)

    # Perform active tensor operation on GPU to prove CUDA execution
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()

    return {
        "cuda_available": True,
        "device_count": device_count,
        "device_name": device_name,
        "compute_capability": f"{device_capability[0]}.{device_capability[1]}",
        "total_vram_gb": vram_gb,
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
    }


def train_baseline(
    data_yaml: Path | str = "artifacts/ealpr_yolo_dataset/dataset.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch_size: int = 16,
    seed: int = 42,
    project_dir: Path | str = "artifacts/runs/detect",
    run_name: str = "ealpr_baseline_yolov8n",
) -> Path:
    logger = setup_logger("train_baseline")
    logger.info("Verifying hardware and CUDA compute before training...")

    gpu_info = verify_gpu_cuda()
    logger.info(
        f"CUDA Verified: {gpu_info['device_name']} ({gpu_info['total_vram_gb']} GB VRAM, "
        f"PyTorch {gpu_info['pytorch_version']}, CUDA {gpu_info['cuda_runtime_version']})"
    )

    data_path = Path(data_yaml).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    from ultralytics import YOLO
    import ultralytics
    import numpy as np

    logger.info(f"Loading base architecture and pretrained weights: {model_name}")
    model = YOLO(model_name)

    project_path = Path(project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initiating training run '{run_name}' for {epochs} epochs (imgsz={imgsz}, batch={batch_size}, seed={seed})...")
    start_time = time.time()
    start_dt = datetime.now(timezone.utc).isoformat()

    # Train model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,  # Explicitly GPU 0
        workers=2,
        seed=seed,
        deterministic=True,
        project=str(project_path),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
    )

    duration_sec = round(time.time() - start_time, 2)
    end_dt = datetime.now(timezone.utc).isoformat()
    logger.info(f"Training completed successfully in {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes).")

    # Locate best checkpoint
    run_dir = project_path / run_name
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    if not best_weights.is_file():
        raise RuntimeError(f"Expected best checkpoint was not created at {best_weights}")

    # Build and save reproducible run summary
    summary = {
        "model": model_name,
        "run_name": run_name,
        "data_config": str(data_path),
        "start_time_utc": start_dt,
        "end_time_utc": end_dt,
        "duration_seconds": duration_sec,
        "hyperparameters": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "seed": seed,
            "deterministic": True,
            "workers": 2,
            "device": "cuda:0",
        },
        "hardware": gpu_info,
        "dependencies": {
            "ultralytics": ultralytics.__version__,
            "torch": gpu_info["pytorch_version"],
            "numpy": np.__version__,
            "python": sys.version,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "best_checkpoint": str(best_weights),
            "last_checkpoint": str(last_weights),
            "results_csv": str(run_dir / "results.csv"),
        },
    }

    summary_file = run_dir / "training_run_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved complete reproducible run summary to: {summary_file}")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline license plate detector.")
    parser.add_argument("--data", type=str, default="artifacts/ealpr_yolo_dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--project", type=str, default="artifacts/runs/detect", help="Project save directory")
    parser.add_argument("--name", type=str, default="ealpr_baseline_yolov8n", help="Run identifier")
    args = parser.parse_args()

    train_baseline(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        seed=args.seed,
        project_dir=args.project,
        run_name=args.name,
    )
