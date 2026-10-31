"""Controlled Preprocessing & Inference Domain-Shift Experiments for Mission 011.

Tests image transformations and inference configurations on a stratified representative
subset of gate captures (Night IR, Evening IR, Dawn, Daylight).
Changes one variable at a time and measures detection count, confidence, and latency.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")
from vehicle_intelligence.pipeline.media_loader import MediaLoader


# Preprocessing functions (one variable changed at a time)
def preproc_raw(img: np.ndarray) -> np.ndarray:
    return img


def preproc_clahe_clip2(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(gray)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)


def preproc_clahe_clip4(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    res = clahe.apply(gray)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)


def preproc_gamma_05(img: np.ndarray) -> np.ndarray:
    # Lift shadows (gamma = 0.5)
    lut = np.array([((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, lut)


def preproc_gamma_07(img: np.ndarray) -> np.ndarray:
    lut = np.array([((i / 255.0) ** 0.7) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, lut)


def preproc_gamma_15(img: np.ndarray) -> np.ndarray:
    # Darken highlights
    lut = np.array([((i / 255.0) ** 1.5) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, lut)


def preproc_autocontrast(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p1, p99 = np.percentile(gray, (1, 99))
    stretched = np.clip((gray - p1) / max(1.0, (p99 - p1)) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)


def preproc_bilateral(img: np.ndarray) -> np.ndarray:
    # Edge-preserving smoothing
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def run_experiments():
    print("=" * 80)
    print("MISSION 011: CONTROLLED PREPROCESSING & RESOLUTION EXPERIMENTS")
    print("=" * 80)

    out_dir = Path("artifacts/missions/mission_011")
    vis_dir = out_dir / "representative_detection_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Representative Subset (16 diverse frames)
    # 5 Deep Night IR, 5 Evening IR, 2 Dawn/Twilight, 4 Daylight
    all_files = sorted(Path("D:/TruckEntry").glob("*.jpg"))
    target_stems = [
        # Night IR (01:39 to 06:05)
        "20261015_013942_111",
        "20261015_030516_425",
        "20261015_031837_578",
        "20261015_041510_522",
        "20261015_051657_638",
        # Dawn (06:30 to 06:45)
        "20261015_063257_536",
        "20261015_064503_984",
        # Daylight (07:00 to 12:00)
        "20261015_070935_483",
        "20261015_071152_587",
        "20261015_081420_532",
        "20261015_120110_340",
        # Evening IR (18:00 to 22:35)
        "20261015_181512_230",
        "20261015_193022_410",
        "20261015_204510_780",
        "20261015_214530_512",
        "20261015_223517_659",
    ]

    selected_files = []
    for stem in target_stems:
        matched = [f for f in all_files if stem in f.name]
        if matched:
            selected_files.append(matched[0])
        else:
            # Fallback to closest file
            prefix = stem[:13]
            matched_p = [f for f in all_files if prefix in f.name]
            if matched_p:
                selected_files.append(matched_p[0])

    # Deduplicate and ensure at least 15 frames
    selected_files = list(dict.fromkeys(selected_files))
    if len(selected_files) < 15:
        # Add additional evening and night frames
        selected_files.extend(all_files[:5])
        selected_files.extend(all_files[-5:])
        selected_files = list(dict.fromkeys(selected_files))[:16]

    print(f"Selected {len(selected_files)} representative stratified gate frames:")
    for f in selected_files:
        print(f"  - {f.name[:35]}")

    # 2. Experimental Configurations: Testing One Variable at a Time
    experiments = [
        # Baseline
        {"name": "Exp01_Baseline_Raw_640", "func": preproc_raw, "imgsz": 640, "conf": 0.20, "desc": "Raw BGR at default imgsz=640"},
        # Resolution variations (unaltered raw)
        {"name": "Exp02_Raw_Resolution_1280", "func": preproc_raw, "imgsz": 1280, "conf": 0.20, "desc": "Raw BGR scaled to 1280px inference size"},
        {"name": "Exp03_Raw_Resolution_1920", "func": preproc_raw, "imgsz": 1920, "conf": 0.20, "desc": "Raw BGR scaled to 1920px inference size"},
        # Confidence threshold variations
        {"name": "Exp04_Raw_Conf_010", "func": preproc_raw, "imgsz": 640, "conf": 0.10, "desc": "Raw BGR with lower confidence threshold (0.10)"},
        {"name": "Exp05_Raw_Conf_005", "func": preproc_raw, "imgsz": 640, "conf": 0.05, "desc": "Raw BGR with very low confidence threshold (0.05)"},
        # CLAHE variations
        {"name": "Exp06_CLAHE_Clip2_640", "func": preproc_clahe_clip2, "imgsz": 640, "conf": 0.20, "desc": "CLAHE with clipLimit=2.0, tileGrid=(8,8)"},
        {"name": "Exp07_CLAHE_Clip4_640", "func": preproc_clahe_clip4, "imgsz": 640, "conf": 0.20, "desc": "CLAHE with clipLimit=4.0, tileGrid=(8,8)"},
        {"name": "Exp08_CLAHE_Clip2_1280", "func": preproc_clahe_clip2, "imgsz": 1280, "conf": 0.20, "desc": "CLAHE clip=2.0 combined with 1280px resolution"},
        # Gamma variations
        {"name": "Exp09_Gamma_05_640", "func": preproc_gamma_05, "imgsz": 640, "conf": 0.20, "desc": "Gamma correction gamma=0.5 (lifts dark shadow tones)"},
        {"name": "Exp10_Gamma_07_640", "func": preproc_gamma_07, "imgsz": 640, "conf": 0.20, "desc": "Gamma correction gamma=0.7 (moderate shadow lift)"},
        {"name": "Exp11_Gamma_15_640", "func": preproc_gamma_15, "imgsz": 640, "conf": 0.20, "desc": "Gamma correction gamma=1.5 (darkens washed highlights)"},
        # Dynamic Range & Denoising
        {"name": "Exp12_AutoContrast_640", "func": preproc_autocontrast, "imgsz": 640, "conf": 0.20, "desc": "Min-Max linear contrast stretch (1st to 99th percentile)"},
        {"name": "Exp13_BilateralFilter_640", "func": preproc_bilateral, "imgsz": 640, "conf": 0.20, "desc": "Bilateral filter (edge-preserving denoising)"},
    ]

    model = YOLO("yolov8n.pt")
    coco_vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    csv_rows = []

    print("\nExecuting controlled experiments...")
    for exp in experiments:
        exp_name = exp["name"]
        exp_func = exp["func"]
        imgsz = exp["imgsz"]
        conf = exp["conf"]
        desc = exp["desc"]

        print(f"\n--- Running: {exp_name} ({desc}) ---")
        total_dets = 0
        night_dets = 0
        day_dets = 0
        conf_scores = []
        latencies = []
        classes_detected = {}

        for f in selected_files:
            fname = f.name
            is_night = any(k in fname for k in ["0139", "0305", "0318", "0415", "0516", "0532", "0605", "18", "19", "20", "21", "22"])
            img = MediaLoader.load_image(f)
            proc_img = exp_func(img)

            t0 = time.perf_counter()
            res = model.predict(
                source=proc_img,
                conf=conf,
                imgsz=imgsz,
                classes=coco_vehicle_classes,
                device="cuda",
                verbose=False,
            )[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat_ms)

            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                c_dets = len(boxes)
                total_dets += c_dets
                if is_night:
                    night_dets += c_dets
                else:
                    day_dets += c_dets

                for b in boxes:
                    c_id = int(b.cls[0])
                    c_name = model.names[c_id]
                    s = float(b.conf[0])
                    conf_scores.append(s)
                    classes_detected[c_name] = classes_detected.get(c_name, 0) + 1

                # Save annotated visual example for representative night/day frames
                if "013942" in fname or "070935" in fname:
                    ann_img = proc_img.copy()
                    for b in boxes:
                        bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                        c_name = model.names[int(b.cls[0])]
                        s = float(b.conf[0])
                        cv2.rectangle(ann_img, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                        cv2.putText(ann_img, f"{c_name} {s:.2f}", (bx1, max(25, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    vis_path = vis_dir / f"{exp_name}_{fname[:19]}.jpg"
                    cv2.imwrite(str(vis_path), cv2.resize(ann_img, (960, 540)))

        mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        mean_lat = float(np.mean(latencies))
        frames_tested = len(selected_files)

        print(f"  Summary: Total Detections: {total_dets} (Day: {day_dets}, Night IR: {night_dets})")
        print(f"  Mean Conf: {mean_conf:.3f} | Mean Latency: {mean_lat:.2f} ms | Classes: {classes_detected}")

        csv_rows.append({
            "experiment_id": exp_name,
            "description": desc,
            "inference_size": imgsz,
            "confidence_threshold": conf,
            "total_frames_tested": frames_tested,
            "total_vehicles_detected": total_dets,
            "daylight_vehicles_detected": day_dets,
            "night_ir_vehicles_detected": night_dets,
            "mean_confidence": round(mean_conf, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "detected_classes": ";".join(f"{k}:{v}" for k, v in classes_detected.items()) if classes_detected else "NONE",
            "night_ir_improvement": "YES" if night_dets > 0 else "NO",
        })

    csv_path = out_dir / "preprocessing_experiments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nExported preprocessing experiments to: {csv_path}")


if __name__ == "__main__":
    run_experiments()
