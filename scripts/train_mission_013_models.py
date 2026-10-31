"""Mission 013: Model Training Script for Track A (RT-DETR-L) and Track B (YOLOv8m).

Executes conservative transfer learning on 196 training frames, using 72 validation frames
for checkpoint selection and early stopping. development_test_v1 is strictly untouched.
Saves best checkpoints, logs, and configuration YAMLs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time
import torch
from ultralytics import RTDETR, YOLO
import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = ROOT_DIR / "artifacts" / "missions" / "mission_013" / "dataset" / "dataset.yaml"
OUT_BASE = ROOT_DIR / "artifacts" / "missions" / "mission_013"
CONFIG_DIR = OUT_BASE / "training_configurations"
LOG_DIR = OUT_BASE / "training_logs"
CKPT_DIR = OUT_BASE / "checkpoints"

RTDETR_PRETRAINED = ROOT_DIR / "rtdetr-l.pt"
YOLOV8M_PRETRAINED = ROOT_DIR / "yolov8m.pt"


def train_track_b_yolov8m():
    print("\n" + "=" * 80)
    print("MISSION 013 - TRACK B: YOLOv8m DOMAIN FINE-TUNING")
    print("=" * 80)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": "yolov8m_finetuned",
        "base_weights": "yolov8m.pt",
        "data": str(DATASET_YAML),
        "epochs": 25,
        "batch": 8,
        "imgsz": 640,
        "device": "0",
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 3,
        "patience": 8,
        "seed": 42,
        "deterministic": True,
        "val": True,
        "project": str(OUT_BASE / "runs" / "yolov8m"),
        "name": "yolov8m_domain_exp",
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "verbose": True,
    }

    config_path = CONFIG_DIR / "yolov8m_train_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Saved configuration to: {config_path}")

    # Initialize model from pretrained weights (preserved untouched)
    model = YOLO(str(YOLOV8M_PRETRAINED))

    t0 = time.time()
    results = model.train(
        data=str(DATASET_YAML),
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=config["device"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        warmup_epochs=config["warmup_epochs"],
        patience=config["patience"],
        seed=config["seed"],
        deterministic=config["deterministic"],
        val=True,
        project=config["project"],
        name=config["name"],
        exist_ok=True,
        save=True,
        verbose=True,
    )
    duration = time.time() - t0
    print(f"YOLOv8m training completed in {duration:.1f}s")

    # Copy best checkpoint
    run_dir = Path(config["project"]) / config["name"]
    best_weights = run_dir / "weights" / "best.pt"
    target_best = CKPT_DIR / "yolov8m_finetuned_best.pt"
    if best_weights.is_file():
        shutil.copy2(best_weights, target_best)
        print(f"Copied best checkpoint to: {target_best}")

    # Copy csv training log
    results_csv = run_dir / "results.csv"
    target_csv = LOG_DIR / "yolov8m_training_results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, target_csv)
        print(f"Copied training results log to: {target_csv}")

    return {
        "model_id": "yolov8m_finetuned",
        "training_time_sec": round(duration, 2),
        "best_checkpoint": str(target_best),
        "run_dir": str(run_dir),
    }


def train_track_a_rtdetr_l():
    print("\n" + "=" * 80)
    print("MISSION 013 - TRACK A: RT-DETR-L DOMAIN FINE-TUNING")
    print("=" * 80)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": "rtdetr_l_finetuned",
        "base_weights": "rtdetr-l.pt",
        "data": str(DATASET_YAML),
        "epochs": 25,
        "batch": 4,  # conservative batch size for transformer memory stability
        "imgsz": 640,
        "device": "0",
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": 0.0001,  # 1e-4 conservative learning rate for transformer
        "lrf": 0.01,
        "warmup_epochs": 3,
        "patience": 8,
        "seed": 42,
        "deterministic": True,
        "val": True,
        "project": str(OUT_BASE / "runs" / "rtdetr_l"),
        "name": "rtdetr_l_domain_exp",
        "exist_ok": True,
        "save": True,
        "save_period": -1,
        "verbose": True,
    }

    config_path = CONFIG_DIR / "rtdetr_l_train_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Saved configuration to: {config_path}")

    # Initialize model from pretrained weights (preserved untouched)
    model = RTDETR(str(RTDETR_PRETRAINED))

    t0 = time.time()
    results = model.train(
        data=str(DATASET_YAML),
        epochs=config["epochs"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        device=config["device"],
        workers=config["workers"],
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        warmup_epochs=config["warmup_epochs"],
        patience=config["patience"],
        seed=config["seed"],
        deterministic=config["deterministic"],
        val=True,
        project=config["project"],
        name=config["name"],
        exist_ok=True,
        save=True,
        verbose=True,
    )
    duration = time.time() - t0
    print(f"RT-DETR-L training completed in {duration:.1f}s")

    # Copy best checkpoint
    run_dir = Path(config["project"]) / config["name"]
    best_weights = run_dir / "weights" / "best.pt"
    target_best = CKPT_DIR / "rtdetr_l_finetuned_best.pt"
    if best_weights.is_file():
        shutil.copy2(best_weights, target_best)
        print(f"Copied best checkpoint to: {target_best}")

    # Copy csv training log
    results_csv = run_dir / "results.csv"
    target_csv = LOG_DIR / "rtdetr_l_training_results.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, target_csv)
        print(f"Copied training results log to: {target_csv}")

    return {
        "model_id": "rtdetr_l_finetuned",
        "training_time_sec": round(duration, 2),
        "best_checkpoint": str(target_best),
        "run_dir": str(run_dir),
    }


def main():
    print("=" * 80)
    print("MISSION 013: EXECUTING DOMAIN FINE-TUNING EXPERIMENTS")
    print("=" * 80)

    # Track B: YOLOv8m
    yolo_summary = train_track_b_yolov8m()

    # Track A: RT-DETR-L
    rtdetr_summary = train_track_a_rtdetr_l()

    summary_manifest = {
        "milestone": "MISSION_013",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": {
            "track_a_rtdetr_l": rtdetr_summary,
            "track_b_yolov8m": yolo_summary,
        }
    }

    manifest_path = OUT_BASE / "training_summary_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2, ensure_ascii=False)
    print(f"\nAll training runs complete. Summary manifest: {manifest_path}")


if __name__ == "__main__":
    main()
