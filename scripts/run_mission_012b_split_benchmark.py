"""Mission 012B: Split-Specific Benchmark Evaluation Script.

Evaluates YOLOv8n, YOLOv8m, and RT-DETR-L across:
1. Training Set (196 frames)
2. Validation Set (72 frames)
3. Held-Out Test Set (72 frames) -> Used for official model comparison
4. Entire Dataset (340 frames, explicitly labeled descriptive / in-sample)

Exports:
- artifacts/missions/mission_012b/split_specific_benchmark.csv
- artifacts/missions/mission_012b/per_lighting_test_metrics.csv
- artifacts/missions/mission_012b/per_class_test_metrics.csv
- artifacts/missions/mission_012b/per_visibility_test_metrics.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")
from vehicle_intelligence.pipeline.media_loader import MediaLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path("D:/TruckEntry")
OUT_DIR = ROOT_DIR / "artifacts" / "missions" / "mission_012b"
GROUND_TRUTH_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "vehicle_ground_truth.json"
BENCHMARK_SPLITS_FILE = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"

SPLIT_BENCHMARK_CSV = OUT_DIR / "split_specific_benchmark.csv"
PER_LIGHTING_TEST_CSV = OUT_DIR / "per_lighting_test_metrics.csv"
PER_CLASS_TEST_CSV = OUT_DIR / "per_class_test_metrics.csv"
PER_VISIBILITY_TEST_CSV = OUT_DIR / "per_visibility_test_metrics.csv"

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def match_detections(gt_boxes: List[Dict], pred_boxes: List[Dict], iou_threshold: float = 0.50) -> Tuple[int, int, int]:
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    matched_pred = set()

    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1]["confidence"], reverse=True)

    for p_idx, p in sorted_preds:
        p_box = np.array(p["bbox_xyxy"])
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, g in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            g_box = np.array(g["bbox_xyxy"])
            iou = box_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)

    tp = len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def run_split_benchmark():
    print("=" * 80)
    print("MISSION 012B: RIGOROUS SPLIT-SPECIFIC VEHICLE DETECTOR BENCHMARK")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth: Dict[str, Dict] = json.load(f)

    with open(BENCHMARK_SPLITS_FILE, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    train_filenames: Set[str] = set(splits_data["train"]["filenames"])
    val_filenames: Set[str] = set(splits_data["val"]["filenames"])
    test_filenames: Set[str] = set(splits_data["test"]["filenames"])

    print(f"Splits loaded: Train={len(train_filenames)}, Val={len(val_filenames)}, Test={len(test_filenames)}")

    models_to_test = [
        {"model_id": "yolov8n", "weights": "yolov8n.pt", "architecture": "YOLOv8 Nano (CNN)", "params_m": 3.2, "license": "AGPL-3.0"},
        {"model_id": "yolov8m", "weights": "yolov8m.pt", "architecture": "YOLOv8 Medium (CNN)", "params_m": 25.9, "license": "AGPL-3.0"},
        {"model_id": "rtdetr-l", "weights": "rtdetr-l.pt", "architecture": "RT-DETR Large (Transformer)", "params_m": 32.0, "license": "Apache-2.0"},
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    all_files = sorted(list(GATE_DIR.glob("*.jpg")))
    iou_thresholds = np.linspace(0.50, 0.95, 10)

    # 1. Store model predictions per file
    model_predictions: Dict[str, Dict[str, List[Dict]]] = {}
    model_latencies: Dict[str, Dict[str, float]] = {}
    checkpoint_hashes: Dict[str, str] = {}

    for m_info in models_to_test:
        model_id = m_info["model_id"]
        weights_path = Path(m_info["weights"])
        checkpoint_hashes[model_id] = compute_sha256(weights_path)
        print(f"\nRunning Inference: {model_id} (SHA-256: {checkpoint_hashes[model_id][:12]}...)...")

        model = YOLO(str(weights_path))

        # Warm-up (3 dummy iterations)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _ = model.predict(dummy, device=device, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()

        model_predictions[model_id] = {}
        model_latencies[model_id] = {}

        for f in all_files:
            img = MediaLoader.load_image(f)
            if img is None:
                continue

            if device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                res = model.predict(img, conf=0.25, classes=[2, 3, 5, 7], imgsz=640, device=device, verbose=False)[0]
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                res = model.predict(img, conf=0.25, classes=[2, 3, 5, 7], imgsz=640, device=device, verbose=False)[0]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            model_latencies[model_id][f.name] = elapsed_ms

            preds = []
            if res.boxes is not None and len(res.boxes) > 0:
                p_boxes = res.boxes.xyxy.cpu().numpy()
                p_confs = res.boxes.conf.cpu().numpy()
                p_clss = res.boxes.cls.cpu().numpy().astype(int)

                # NMS deduplication
                if len(p_boxes) > 1:
                    indices = cv2.dnn.NMSBoxes(
                        bboxes=[[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in p_boxes],
                        scores=[float(s) for s in p_confs],
                        score_threshold=0.25,
                        nms_threshold=0.50,
                    )
                    if len(indices) > 0:
                        keep_idx = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indices]
                        p_boxes = p_boxes[keep_idx]
                        p_confs = p_confs[keep_idx]
                        p_clss = p_clss[keep_idx]

                for b, score, cid in zip(p_boxes, p_confs, p_clss):
                    preds.append({
                        "bbox_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "confidence": float(score),
                        "class": COCO_VEHICLE_CLASSES.get(cid, "car"),
                    })

            model_predictions[model_id][f.name] = preds

    # 2. Compute Metrics Separately Across Splits
    split_definitions = {
        "held_out_test_set": test_filenames,
        "validation_set": val_filenames,
        "training_set": train_filenames,
        "entire_dataset_in_sample_descriptive": set(f.name for f in all_files),
    }

    benchmark_rows = []

    for m_info in models_to_test:
        model_id = m_info["model_id"]

        for split_name, split_files in split_definitions.items():
            tp_50 = 0
            fp_50 = 0
            fn_50 = 0
            frames_with_detection = 0
            total_pred_boxes = 0
            total_gt_boxes = 0

            # mAP accumulators
            ap_list = []
            tp_by_th = {th: 0 for th in iou_thresholds}
            fp_by_th = {th: 0 for th in iou_thresholds}
            fn_by_th = {th: 0 for th in iou_thresholds}

            latencies = []

            for fname in split_files:
                # Find sample ID
                sample_id = None
                for sid, entry in ground_truth.items():
                    if entry["filename"] == fname:
                        sample_id = sid
                        break
                gt_boxes = ground_truth[sample_id]["boxes"] if sample_id else []
                total_gt_boxes += len(gt_boxes)

                preds = model_predictions[model_id].get(fname, [])
                total_pred_boxes += len(preds)
                if len(preds) > 0:
                    frames_with_detection += 1

                lat = model_latencies[model_id].get(fname, 0.0)
                latencies.append(lat)

                tp, fp, fn = match_detections(gt_boxes, preds, iou_threshold=0.50)
                tp_50 += tp
                fp_50 += fp
                fn_50 += fn

                for th in iou_thresholds:
                    tp_th, fp_th, fn_th = match_detections(gt_boxes, preds, iou_threshold=th)
                    tp_by_th[th] += tp_th
                    fp_by_th[th] += fp_th
                    fn_by_th[th] += fn_th

            precision = tp_50 / max(1, (tp_50 + fp_50))
            recall = tp_50 / max(1, (tp_50 + fn_50))
            f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

            for th in iou_thresholds:
                p_th = tp_by_th[th] / max(1, (tp_by_th[th] + fp_by_th[th]))
                r_th = tp_by_th[th] / max(1, (tp_by_th[th] + fn_by_th[th]))
                ap_list.append(p_th * r_th)
            map_50_95 = float(np.mean(ap_list))

            mean_lat = float(np.mean(latencies)) if latencies else 0.0
            fp_per_img = fp_50 / max(1, len(split_files))
            frame_yield = (frames_with_detection / max(1, len(split_files))) * 100.0

            benchmark_rows.append({
                "model_id": model_id,
                "architecture": m_info["architecture"],
                "checkpoint_sha256": checkpoint_hashes[model_id][:16],
                "dataset_split": split_name,
                "frame_count": len(split_files),
                "ground_truth_boxes": total_gt_boxes,
                "predicted_boxes": total_pred_boxes,
                "true_positives_50": tp_50,
                "false_positives_50": fp_50,
                "false_negatives_50": fn_50,
                "precision_50": round(precision, 4),
                "recall_50": round(recall, 4),
                "f1_score_50": round(f1, 4),
                "mAP_50": round(precision, 4),
                "mAP_50_95": round(map_50_95, 4),
                "false_positives_per_image": round(fp_per_img, 4),
                "frame_detection_yield_pct": round(frame_yield, 2),
                "mean_gpu_latency_ms": round(mean_lat, 2),
                "throughput_fps": round(1000.0 / max(0.1, mean_lat), 1),
                "conf_threshold": 0.25,
                "iou_matching_threshold": 0.50,
            })

    # Save split_specific_benchmark.csv
    with open(SPLIT_BENCHMARK_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"\nSaved split-specific benchmark to: {SPLIT_BENCHMARK_CSV}")

    # 3. Compute HELD-OUT TEST SET Granular Breakdowns
    # 3.1 Per-Lighting Test Metrics
    lighting_test_rows = []
    lighting_groups = {
        "NIGHT_IR": [fn for fn in test_filenames if ground_truth_lighting(fn, ground_truth) == "NIGHT_IR"],
        "DAYLIGHT": [fn for fn in test_filenames if ground_truth_lighting(fn, ground_truth) == "DAYLIGHT"],
        "EVENING_IR": [fn for fn in test_filenames if ground_truth_lighting(fn, ground_truth) == "EVENING_IR"],
    }

    for m_info in models_to_test:
        model_id = m_info["model_id"]
        for regime, fnames in lighting_groups.items():
            tp_reg, fp_reg, fn_reg = 0, 0, 0
            det_frames = 0
            total_gt = 0

            for fn in fnames:
                sid = get_sample_id_from_filename(fn, ground_truth)
                gt_boxes = ground_truth[sid]["boxes"] if sid else []
                total_gt += len(gt_boxes)

                preds = model_predictions[model_id].get(fn, [])
                if len(preds) > 0:
                    det_frames += 1

                tp, fp, fn_m = match_detections(gt_boxes, preds, iou_threshold=0.50)
                tp_reg += tp
                fp_reg += fp
                fn_reg += fn_m

            prec = tp_reg / max(1, (tp_reg + fp_reg))
            rec = tp_reg / max(1, (tp_reg + fn_reg))
            f1 = (2 * prec * rec) / max(1e-6, (prec + rec))
            yield_pct = (det_frames / max(1, len(fnames))) * 100.0

            lighting_test_rows.append({
                "model_id": model_id,
                "dataset_split": "held_out_test_set",
                "lighting_regime": regime,
                "test_frame_count": len(fnames),
                "sample_size_disclosure": f"n={len(fnames)} frames (statistically limited for NIGHT_IR)" if regime == "NIGHT_IR" else f"n={len(fnames)} frames",
                "ground_truth_boxes": total_gt,
                "frames_with_detection": det_frames,
                "frame_detection_yield_pct": round(yield_pct, 2),
                "true_positives": tp_reg,
                "false_positives": fp_reg,
                "false_negatives": fn_reg,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            })

    with open(PER_LIGHTING_TEST_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(lighting_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(lighting_test_rows)
    print(f"Saved per-lighting test metrics to: {PER_LIGHTING_TEST_CSV}")

    # 3.2 Per-Class Test Metrics
    class_test_rows = []
    # Collect all GT boxes in test set
    test_gt_boxes_by_class: Dict[str, int] = {}
    for fn in test_filenames:
        sid = get_sample_id_from_filename(fn, ground_truth)
        for b in ground_truth[sid]["boxes"]:
            cls = b["class"]
            test_gt_boxes_by_class[cls] = test_gt_boxes_by_class.get(cls, 0) + 1

    for m_info in models_to_test:
        model_id = m_info["model_id"]
        for cls, count in test_gt_boxes_by_class.items():
            # Measure recall on this class in test split
            recalled = 0
            for fn in test_filenames:
                sid = get_sample_id_from_filename(fn, ground_truth)
                target_gt = [b for b in ground_truth[sid]["boxes"] if b["class"] == cls]
                preds = model_predictions[model_id].get(fn, [])
                tp, _, _ = match_detections(target_gt, preds, iou_threshold=0.50)
                recalled += tp

            rec = recalled / max(1, count)
            class_test_rows.append({
                "model_id": model_id,
                "dataset_split": "held_out_test_set",
                "ground_truth_class": cls,
                "test_instance_count": count,
                "sample_size_note": f"n={count} instances in held-out test split",
                "recalled_instances": recalled,
                "class_recall_50": round(rec, 4),
            })

    with open(PER_CLASS_TEST_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_test_rows)
    print(f"Saved per-class test metrics to: {PER_CLASS_TEST_CSV}")

    # 3.3 Per-Visibility Test Metrics
    visibility_test_rows = []
    vis_counts: Dict[str, int] = {}
    for fn in test_filenames:
        sid = get_sample_id_from_filename(fn, ground_truth)
        for b in ground_truth[sid]["boxes"]:
            v = b["visibility"]
            vis_counts[v] = vis_counts.get(v, 0) + 1

    for m_info in models_to_test:
        model_id = m_info["model_id"]
        for vis_cat, count in vis_counts.items():
            recalled = 0
            for fn in test_filenames:
                sid = get_sample_id_from_filename(fn, ground_truth)
                target_gt = [b for b in ground_truth[sid]["boxes"] if b["visibility"] == vis_cat]
                preds = model_predictions[model_id].get(fn, [])
                tp, _, _ = match_detections(target_gt, preds, iou_threshold=0.50)
                recalled += tp

            rec = recalled / max(1, count)
            visibility_test_rows.append({
                "model_id": model_id,
                "dataset_split": "held_out_test_set",
                "visibility_category": vis_cat,
                "test_instance_count": count,
                "recalled_instances": recalled,
                "recall_50": round(rec, 4),
            })

    with open(PER_VISIBILITY_TEST_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(visibility_test_rows[0].keys()))
        writer.writeheader()
        writer.writerows(visibility_test_rows)
    print(f"Saved per-visibility test metrics to: {PER_VISIBILITY_TEST_CSV}")


def ground_truth_lighting(filename: str, ground_truth: Dict[str, Dict]) -> str:
    for sid, entry in ground_truth.items():
        if entry["filename"] == filename:
            return entry["lighting_condition"]
    return "DAYLIGHT"


def get_sample_id_from_filename(filename: str, ground_truth: Dict[str, Dict]) -> Optional[str]:
    for sid, entry in ground_truth.items():
        if entry["filename"] == filename:
            return sid
    return None


if __name__ == "__main__":
    run_split_benchmark()
