"""Evaluation and Comparison of Alternative Pretrained Vehicle Detectors for Mission 011.

Compares YOLOv8n, YOLOv8s, YOLOv8m, YOLO11n, and RT-DETR-L on representative gate captures.
Records performance metrics, latency, license, and outputs vehicle_detector_comparison.csv.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path
import sys
import time
from typing import Dict, List
import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")
from vehicle_intelligence.pipeline.media_loader import MediaLoader


def run_model_comparison():
    print("=" * 80)
    print("MISSION 011: ALTERNATIVE PRETRAINED VEHICLE DETECTOR EVALUATION")
    print("=" * 80)

    out_dir = Path("artifacts/missions/mission_011")
    vis_dir = out_dir / "representative_detection_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stratified Representative Test Set (16 frames)
    all_files = sorted(Path("D:/TruckEntry").glob("*.jpg"))
    target_patterns = [
        # Night IR
        "*013942*", "*030516*", "*031837*", "*041510*", "*051657*", "*053247*", "*060530*",
        # Dawn
        "*063257*", "*064503*",
        # Daylight
        "*070935*", "*071152*", "*081420*", "*120144*",
        # Evening IR
        "*193049*", "*204507*", "*223517*",
    ]

    selected_files = []
    for pat in target_patterns:
        m = list(Path("D:/TruckEntry").glob(pat))
        if m:
            selected_files.append(m[0])
    selected_files = list(dict.fromkeys(selected_files))[:16]

    print(f"Loaded {len(selected_files)} representative benchmark frames.")

    # 2. Candidate Models Specification
    models_config = [
        {
            "model_id": "yolov8n",
            "weights": "yolov8n.pt",
            "architecture": "YOLOv8 Nano (Anchor-free CNN)",
            "license": "AGPL-3.0",
            "commercial_compatible": "No (Requires Commercial License from Ultralytics or Open Source Disclosure)",
            "params_m": 3.2,
        },
        {
            "model_id": "yolov8s",
            "weights": "yolov8s.pt",
            "architecture": "YOLOv8 Small (CNN)",
            "license": "AGPL-3.0",
            "commercial_compatible": "No (Requires Commercial License)",
            "params_m": 11.2,
        },
        {
            "model_id": "yolov8m",
            "weights": "yolov8m.pt",
            "architecture": "YOLOv8 Medium (CNN)",
            "license": "AGPL-3.0",
            "commercial_compatible": "No (Requires Commercial License)",
            "params_m": 25.9,
        },
        {
            "model_id": "yolo11n",
            "weights": "yolo11n.pt",
            "architecture": "YOLO11 Nano (Next-Gen CNN/Attention)",
            "license": "AGPL-3.0",
            "commercial_compatible": "No (Requires Commercial License)",
            "params_m": 2.6,
        },
        {
            "model_id": "rtdetr-l",
            "weights": "rtdetr-l.pt",
            "architecture": "RT-DETR Large (Real-Time Vision Transformer)",
            "license": "Apache-2.0",
            "commercial_compatible": "Yes (Permissive Open Source, Commercial Friendly)",
            "params_m": 32.0,
        },
    ]

    results_rows = []

    for cfg in models_config:
        mid = cfg["model_id"]
        wpath = cfg["weights"]
        arch = cfg["architecture"]
        lic = cfg["license"]
        comm = cfg["commercial_compatible"]
        params = cfg["params_m"]

        print(f"\n--- Evaluating Model: {mid} ({arch}) ---")
        try:
            m = YOLO(wpath)
        except Exception as e:
            print(f"Could not instantiate {wpath}: {e}")
            continue

        total_dets = 0
        night_dets = 0
        day_dets = 0
        conf_scores = []
        latencies = []
        classes_detected = {}

        for f in selected_files:
            fname = f.name
            is_night = any(k in fname for k in ["0139", "0305", "0318", "0415", "0516", "0532", "0605", "1930", "2045", "2235"])
            img = MediaLoader.load_image(f)

            t0 = time.perf_counter()
            # Test at confidence threshold 0.20
            res = m.predict(
                source=img,
                conf=0.20,
                imgsz=640,
                classes=[2, 3, 5, 7],  # vehicle classes
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
                    c_name = m.names[c_id]
                    s = float(b.conf[0])
                    conf_scores.append(s)
                    classes_detected[c_name] = classes_detected.get(c_name, 0) + 1

                # Save representative visual example
                if "070935" in fname or "063257" in fname or ("013942" in fname and len(boxes) > 0):
                    ann_img = img.copy()
                    for b in boxes:
                        bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                        c_name = m.names[int(b.cls[0])]
                        s = float(b.conf[0])
                        cv2.rectangle(ann_img, (bx1, by1), (bx2, by2), (0, 220, 0), 3)
                        cv2.putText(ann_img, f"{c_name} {s:.2f}", (bx1, max(30, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
                    vis_path = vis_dir / f"model_comp_{mid}_{fname[:19]}.jpg"
                    cv2.imwrite(str(vis_path), cv2.resize(ann_img, (960, 540)))

        mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        mean_lat = float(np.mean(latencies))
        frames_count = len(selected_files)

        print(f"  Total Detections: {total_dets} (Day: {day_dets}, Night IR: {night_dets})")
        print(f"  Mean Confidence: {mean_conf:.3f} | Latency: {mean_lat:.2f} ms | Classes: {classes_detected}")

        results_rows.append({
            "model_id": mid,
            "architecture": arch,
            "weights_file": wpath,
            "parameters_millions": params,
            "license": lic,
            "commercial_compatible": comm,
            "total_benchmark_frames": frames_count,
            "total_vehicles_detected": total_dets,
            "daylight_vehicles_detected": day_dets,
            "night_ir_vehicles_detected": night_dets,
            "night_detection_rate_pct": round(night_dets / 10.0 * 100.0, 1),
            "mean_confidence": round(mean_conf, 4),
            "mean_latency_ms": round(mean_lat, 2),
            "detected_classes": ";".join(f"{k}:{v}" for k, v in classes_detected.items()) if classes_detected else "NONE",
        })

    csv_path = out_dir / "vehicle_detector_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nExported vehicle detector comparison to: {csv_path}")


if __name__ == "__main__":
    run_model_comparison()
