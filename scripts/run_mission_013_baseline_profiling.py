"""Mission 013: Pre-Training Baseline Profiling and Metric Recording.

Evaluates pretrained RT-DETR-L and YOLOv8m on development_test_v1 (72 frames)
and Validation split (72 frames). Profiles detector latency with CUDA synchronization
(50 warm-up, 200 timed runs) and GPU memory. Records cryptographic hashes.
Outputs: artifacts/missions/mission_013/baseline_metrics.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path("D:/TruckEntry")
SPLITS_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
GT_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "vehicle_ground_truth.json"
DATASET_YAML = ROOT_DIR / "artifacts" / "missions" / "mission_013" / "dataset" / "dataset.yaml"
OUT_JSON = ROOT_DIR / "artifacts" / "missions" / "mission_013" / "baseline_metrics.json"

RTDETR_WEIGHTS = ROOT_DIR / "rtdetr-l.pt"
YOLOV8M_WEIGHTS = ROOT_DIR / "yolov8m.pt"

COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0


def match_detections(pred_boxes, gt_boxes, iou_thresh: float = 0.50) -> Tuple[int, int, int]:
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0
    if not pred_boxes and gt_boxes:
        return 0, 0, len(gt_boxes)
    if pred_boxes and not gt_boxes:
        return 0, len(pred_boxes), 0

    matched_gt = set()
    matched_pred = set()

    # Sort predicted by confidence descending
    sorted_preds = sorted(pred_boxes, key=lambda b: b.get("conf", 0.0), reverse=True)

    tp = 0
    for p_idx, p in enumerate(sorted_preds):
        p_box = p["bbox_xyxy"]
        best_iou = 0.0
        best_gt = -1
        for g_idx, g in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            iou = compute_iou(p_box, g["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx

        if best_iou >= iou_thresh and best_gt != -1:
            tp += 1
            matched_gt.add(best_gt)
            matched_pred.add(p_idx)

    fp = len(sorted_preds) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def profile_detector_latency(model, device_idx: int = 0) -> Dict:
    print("  Profiling detector latency with CUDA synchronization...")
    torch_dev = torch.device(f"cuda:{device_idx}")
    model.to(torch_dev)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_idx)

    dummy_input = torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=torch_dev)

    # 50 warm-up runs
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize(device_idx)

    # 200 timed iterations
    latencies_ms = []
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize(device_idx)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    peak_vram_mib = torch.cuda.max_memory_allocated(device_idx) / (1024.0 * 1024.0)

    latencies_arr = np.array(latencies_ms)
    return {
        "warmup_iterations": 50,
        "timed_iterations": 200,
        "input_resolution": "640x640",
        "device": torch.cuda.get_device_name(device_idx),
        "mean_latency_ms": round(float(np.mean(latencies_arr)), 3),
        "median_p50_latency_ms": round(float(np.median(latencies_arr)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "min_latency_ms": round(float(np.min(latencies_arr)), 3),
        "max_latency_ms": round(float(np.max(latencies_arr)), 3),
        "std_latency_ms": round(float(np.std(latencies_arr)), 3),
        "fps_mean": round(1000.0 / float(np.mean(latencies_arr)), 2),
        "peak_gpu_memory_allocated_mib": round(float(peak_vram_mib), 2),
    }


def evaluate_split(model, split_name: str, split_files: List[str], gt_data: Dict) -> Dict:
    total_frames = len(split_files)
    frames_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_preds = 0
    total_gt = 0

    lighting_stats = {
        "NIGHT_IR": {"frames": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "DAYLIGHT": {"frames": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0, "gt": 0},
        "EVENING_IR": {"frames": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0, "gt": 0},
    }

    class_stats = {
        "heavy_truck": {"gt": 0, "tp": 0},
        "passenger_car": {"gt": 0, "tp": 0},
    }

    visibility_stats = {
        "truncated_by_frame": {"gt": 0, "tp": 0},
        "heavily_occluded": {"gt": 0, "tp": 0},
    }

    for fname in split_files:
        entry = next(v for v in gt_data.values() if v["filename"] == fname)
        gt_boxes = entry["boxes"]
        lighting = entry["lighting_condition"]

        lighting_stats[lighting]["frames"] += 1
        lighting_stats[lighting]["gt"] += len(gt_boxes)
        total_gt += len(gt_boxes)

        for b in gt_boxes:
            c = b["class"]
            if c in class_stats:
                class_stats[c]["gt"] += 1
            vis = b["visibility"]
            if vis in visibility_stats:
                visibility_stats[vis]["gt"] += 1

        img_path = GATE_DIR / fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        res = model.predict(
            source=img,
            conf=0.25,
            imgsz=640,
            classes=COCO_VEHICLE_CLASSES,
            device="cuda:0",
            verbose=False,
        )[0]

        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            frames_detected += 1
            lighting_stats[lighting]["detected"] += 1
            for b in res.boxes:
                coords = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                preds.append({"bbox_xyxy": coords, "conf": conf})

        total_preds += len(preds)
        tp, fp, fn = match_detections(preds, gt_boxes, iou_thresh=0.50)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        lighting_stats[lighting]["tp"] += tp
        lighting_stats[lighting]["fp"] += fp
        lighting_stats[lighting]["fn"] += fn

        # Matched breakdown per instance
        matched_gt_indices = set()
        for p in sorted(preds, key=lambda x: x["conf"], reverse=True):
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                if compute_iou(p["bbox_xyxy"], g["bbox_xyxy"]) >= 0.50:
                    matched_gt_indices.add(g_idx)
                    c = g["class"]
                    if c in class_stats:
                        class_stats[c]["tp"] += 1
                    vis = g["visibility"]
                    if vis in visibility_stats:
                        visibility_stats[vis]["tp"] += 1
                    break

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fp_per_frame = total_fp / total_frames if total_frames > 0 else 0.0
    yield_pct = (frames_detected / total_frames * 100.0) if total_frames > 0 else 0.0

    return {
        "split_name": split_name,
        "total_frames": total_frames,
        "total_ground_truth_vehicles": total_gt,
        "total_predicted_boxes": total_preds,
        "true_positives_50": total_tp,
        "false_positives_50": total_fp,
        "false_negatives_50": total_fn,
        "precision_50": round(prec, 4),
        "recall_50": round(rec, 4),
        "f1_score_50": round(f1, 4),
        "false_positives_per_frame": round(fp_per_frame, 4),
        "frame_detection_yield_pct": round(yield_pct, 2),
        "lighting_breakdown": {
            k: {
                "frames": v["frames"],
                "frames_detected": v["detected"],
                "yield_pct": round(v["detected"] / v["frames"] * 100.0, 2) if v["frames"] > 0 else 0.0,
                "tp": v["tp"],
                "fp": v["fp"],
                "fn": v["fn"],
                "precision": round(v["tp"] / (v["tp"] + v["fp"]), 4) if (v["tp"] + v["fp"]) > 0 else 0.0,
                "recall": round(v["tp"] / v["gt"], 4) if v["gt"] > 0 else 0.0,
            }
            for k, v in lighting_stats.items()
        },
        "class_breakdown": {
            k: {
                "ground_truth_instances": v["gt"],
                "true_positives": v["tp"],
                "recall": round(v["tp"] / v["gt"], 4) if v["gt"] > 0 else 0.0,
            }
            for k, v in class_stats.items()
        },
        "visibility_breakdown": {
            k: {
                "ground_truth_instances": v["gt"],
                "true_positives": v["tp"],
                "recall": round(v["tp"] / v["gt"], 4) if v["gt"] > 0 else 0.0,
            }
            for k, v in visibility_stats.items()
        },
    }


def main():
    print("=" * 80)
    print("MISSION 013: BASELINE PROFILING & PRE-TRAINING METRIC RECORDING")
    print("=" * 80)

    hashes = {
        "rtdetr_l_checkpoint_sha256": sha256_file(RTDETR_WEIGHTS),
        "yolov8m_checkpoint_sha256": sha256_file(YOLOV8M_WEIGHTS),
        "vehicle_ground_truth_sha256": sha256_file(GT_FILE),
        "benchmark_splits_sha256": sha256_file(SPLITS_FILE),
        "dataset_yaml_sha256": sha256_file(DATASET_YAML),
    }

    with open(SPLITS_FILE, "r", encoding="utf-8") as f:
        splits = json.load(f)

    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    test_files = splits["test"]["filenames"]
    val_files = splits["val"]["filenames"]

    # 1. Profile Pretrained RT-DETR-L
    print("\n--- Evaluating Pretrained RT-DETR-L ---")
    rt_model = RTDETR(str(RTDETR_WEIGHTS))
    rt_latency = profile_detector_latency(rt_model.model, device_idx=0)
    print(f"  RT-DETR-L Latency: p50={rt_latency['median_p50_latency_ms']}ms, p95={rt_latency['p95_latency_ms']}ms, Peak VRAM={rt_latency['peak_gpu_memory_allocated_mib']}MiB")

    rt_test_eval = evaluate_split(rt_model, "development_test_v1", test_files, gt_data)
    rt_val_eval = evaluate_split(rt_model, "validation_set", val_files, gt_data)

    # 2. Profile Pretrained YOLOv8m
    print("\n--- Evaluating Pretrained YOLOv8m ---")
    yolo_model = YOLO(str(YOLOV8M_WEIGHTS))
    yolo_latency = profile_detector_latency(yolo_model.model, device_idx=0)
    print(f"  YOLOv8m Latency: p50={yolo_latency['median_p50_latency_ms']}ms, p95={yolo_latency['p95_latency_ms']}ms, Peak VRAM={yolo_latency['peak_gpu_memory_allocated_mib']}MiB")

    yolo_test_eval = evaluate_split(yolo_model, "development_test_v1", test_files, gt_data)
    yolo_val_eval = evaluate_split(yolo_model, "validation_set", val_files, gt_data)

    output = {
        "milestone": "MISSION_013",
        "title": "Pre-Training Reproducible Baseline Metrics and Latency Profiles",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cryptographic_manifest": hashes,
        "experimental_protocol": {
            "evaluation_resolution": "640x640",
            "confidence_threshold": 0.25,
            "iou_matching_threshold": 0.50,
            "latency_warmup_runs": 50,
            "latency_timed_runs": 200,
            "device": "NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)",
        },
        "pretrained_baselines": {
            "rtdetr_l": {
                "checkpoint": str(RTDETR_WEIGHTS.name),
                "sha256": hashes["rtdetr_l_checkpoint_sha256"],
                "architecture": "RT-DETR Large (Vision Transformer, HGNetv2 backbone)",
                "parameters_count": 32970476,
                "license": "Apache-2.0 upstream (Baidu) / AGPL-3.0 Ultralytics wrapper execution",
                "latency_profile": rt_latency,
                "development_test_v1_evaluation": rt_test_eval,
                "validation_evaluation": rt_val_eval,
            },
            "yolov8m": {
                "checkpoint": str(YOLOV8M_WEIGHTS.name),
                "sha256": hashes["yolov8m_checkpoint_sha256"],
                "architecture": "YOLOv8 Medium (CSPDarknet CNN)",
                "parameters_count": 25902640,
                "license": "AGPL-3.0 (Ultralytics)",
                "latency_profile": yolo_latency,
                "development_test_v1_evaluation": yolo_test_eval,
                "validation_evaluation": yolo_val_eval,
            },
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nBaseline metrics successfully recorded to: {OUT_JSON}")


if __name__ == "__main__":
    main()
