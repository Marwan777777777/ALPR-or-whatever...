"""Mission 011 Baseline Evaluation Script.

Evaluates the existing vehicle detector (YOLOv8n COCO, BaseVehicleDetector interface)
across all 340 chronological gate captures using safe MediaLoader unicode decoding.
"""

from __future__ import annotations

import csv
from datetime import datetime
import glob
import os
from pathlib import Path
import re
import sys
import time
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

from vehicle_intelligence.core.domain import BoundingBox, VehicleClass, VehicleDetection
from vehicle_intelligence.detectors.pytorch_yolo import PyTorchYoloVehicleDetector
from vehicle_intelligence.pipeline.media_loader import MediaLoader


def parse_timestamp_from_filename(filename: str) -> Tuple[datetime, float]:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_(\d{3})", filename)
    if not m:
        raise ValueError(f"Could not parse timestamp from {filename}")
    year, month, day, hour, minute, second, ms = [int(v) for v in m.groups()]
    dt = datetime(year, month, day, hour, minute, second, ms * 1000)
    return dt, dt.timestamp()


def categorize_time_of_day(dt: datetime) -> str:
    """Categorizes frame into Night_IR, Daylight, or Dusk/Evening."""
    hour = dt.hour + dt.minute / 60.0
    if hour < 6.5:  # 00:00 to 06:30
        return "NIGHT_IR"
    elif hour < 17.5:  # 06:30 to 17:30
        return "DAYLIGHT"
    else:  # 17:30 to 24:00
        return "EVENING_IR"


def run_baseline_evaluation():
    print("=" * 80)
    print("MISSION 011: BASELINE VEHICLE DETECTION EVALUATION (340 GATE CAPTURES)")
    print("=" * 80)

    out_dir = Path("artifacts/missions/mission_011")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob("D:/TruckEntry/*.jpg"))
    print(f"Discovered {len(files)} gate frames in D:/TruckEntry")

    records = []
    for f in files:
        fname = os.path.basename(f)
        try:
            dt, ts = parse_timestamp_from_filename(fname)
            records.append({
                "path": f,
                "filename": fname,
                "datetime": dt,
                "timestamp": ts,
                "tod": categorize_time_of_day(dt),
            })
        except Exception as e:
            mtime = os.path.getmtime(f)
            dt = datetime.fromtimestamp(mtime)
            records.append({
                "path": f,
                "filename": fname,
                "datetime": dt,
                "timestamp": mtime,
                "tod": categorize_time_of_day(dt),
            })

    records.sort(key=lambda r: r["timestamp"])

    detector = PyTorchYoloVehicleDetector(
        weights_path="yolov8n.pt",
        confidence_threshold=0.20,
        iou_threshold=0.45,
        device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 or os.path.exists("yolov8n.pt") else "cpu",
    )

    rows = []
    latencies = []
    counts_by_tod = {"NIGHT_IR": {"total": 0, "detected": 0}, "DAYLIGHT": {"total": 0, "detected": 0}, "EVENING_IR": {"total": 0, "detected": 0}}
    class_counts = {}

    print("Running baseline inference...")
    for idx, rec in enumerate(records):
        fpath = rec["path"]
        fname = rec["filename"]
        tod = rec["tod"]
        counts_by_tod[tod]["total"] += 1

        img = MediaLoader.load_image(fpath)
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        v_dets = detector.detect(img)
        t_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_ms)

        num_det = len(v_dets)
        if num_det > 0:
            counts_by_tod[tod]["detected"] += 1
            for d in v_dets:
                cls_name = d.vehicle_class.value
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        v_classes = [d.vehicle_class.value for d in v_dets]
        v_confs = [round(d.confidence, 4) for d in v_dets]
        v_boxes = [[int(x) for x in d.box.to_xyxy()] for d in v_dets]

        rows.append({
            "frame_id": idx + 1,
            "filename": fname,
            "timestamp": rec["timestamp"],
            "datetime": rec["datetime"].isoformat(),
            "time_of_day": tod,
            "image_width": w,
            "image_height": h,
            "num_vehicles_detected": num_det,
            "vehicle_classes": ";".join(v_classes) if v_classes else "NONE",
            "confidences": ";".join(str(c) for c in v_confs) if v_confs else "NONE",
            "boxes": ";".join(str(b) for b in v_boxes) if v_boxes else "NONE",
            "inference_latency_ms": round(t_ms, 2),
            "detection_status": "DETECTED" if num_det > 0 else "ZERO_DETECTION",
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(records):
            print(f"Evaluated {idx + 1}/{len(records)} frames...")

    csv_path = out_dir / "vehicle_detection_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported baseline detection results to: {csv_path}")

    # Print summary statistics
    total_frames = len(records)
    total_detected = sum(1 for r in rows if r["num_vehicles_detected"] > 0)
    avg_latency = float(np.mean(latencies))

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY STATISTICS:")
    print(f"Total Frames: {total_frames}")
    print(f"Total Frames with Vehicle Detection: {total_detected}/{total_frames} ({total_detected / total_frames * 100:.2f}%)")
    print(f"Total Frames with ZERO Detection: {total_frames - total_detected}/{total_frames} ({(total_frames - total_detected) / total_frames * 100:.2f}%)")
    print(f"Mean Inference Latency: {avg_latency:.2f} ms")
    print("\nBreakdown by Illumination / Time of Day:")
    for tod, stats in counts_by_tod.items():
        tot = stats["total"]
        det = stats["detected"]
        pct = (det / tot * 100.0) if tot > 0 else 0.0
        print(f"  {tod:<12}: {det}/{tot} frames detected ({pct:.2f}%) | {tot - det} misses")

    print("\nClass Breakdown across All Detections:")
    for cname, ccount in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cname:<12}: {ccount} bounding boxes")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_evaluation()
