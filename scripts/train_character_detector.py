#!/usr/bin/env python3
"""Trains the 27-Class Egyptian License Plate Character Detector on plate crops.

Uses Ultralytics YOLOv8n initialized on GPU, training on the leakage-safe
EALPR character dataset.
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

import torch
from ultralytics import YOLO
from vehicle_intelligence.core.logging import setup_logger


def train_character_model(
    data_yaml: Path | str = "artifacts/ealpr_characters_yolo_dataset/dataset.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 20,
    imgsz: int = 320,
    batch_size: int = 32,
    seed: int = 42,
    project_dir: Path | str = "artifacts/runs/detect_chars",
    run_name: str = "ealpr_characters_yolov8n",
) -> Path:
    logger = setup_logger("train_character_model")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training.")

    logger.info(f"Initializing YOLO character model '{model_name}' on GPU: {torch.cuda.get_device_name(0)}")
    model = YOLO(model_name)

    data_yaml_path = Path(data_yaml).resolve()
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Missing character dataset YAML at {data_yaml_path}")

    start_time = time.time()
    logger.info(f"Starting character model training for {epochs} epochs, imgsz={imgsz}, batch={batch_size}...")

    train_results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,
        seed=seed,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    elapsed_s = round(time.time() - start_time, 2)
    logger.info(f"Character model training completed in {elapsed_s}s ({elapsed_s/60:.2f} mins).")

    run_dir = Path(project_dir) / run_name
    best_weights = run_dir / "weights" / "best.pt"
    logger.info(f"Best character model weights saved to: {best_weights}")
    return best_weights


def main():
    parser = argparse.ArgumentParser(description="Train 27-class plate character detector.")
    parser.add_argument("--data", type=str, default="artifacts/ealpr_characters_yolo_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--name", type=str, default="ealpr_characters_yolov8n")
    args = parser.parse_args()

    train_character_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
