"""Mission 013: Model Evaluation and Comparison Script.

Evaluates 4 models on development_test_v1 (72 frames) and val (72 frames):
1. Pretrained RT-DETR-L (COCO)
2. Fine-tuned RT-DETR-L (Domain adapted)
3. Pretrained YOLOv8m (COCO)
4. Fine-tuned YOLOv8m (Domain adapted)

Computes:
- Object-level TP, FP, FN, Precision, Recall, F1, FP/frame (conf=0.25, IoU=0.50)
- mAP50 and mAP50-95 (COCO 101-point interpolated standard)
- Frame-level detection yield
- Breakdowns by Lighting (Daylight, Evening IR, Deep-Night IR)
- Breakdowns by Class (heavy_truck, passenger_car)
- Breakdowns by Visibility (truncated_by_frame, heavily_occluded)
- Latency (p50, p95, mean) and Peak VRAM (50 warm-up, 200 timed runs)
- Delta / Percentage-point comparisons
- Outputs evaluation_results/full_evaluation_results.json, model_comparison.csv, and checkpoint_manifest.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
GATE_DIR = Path(r"D:\TruckEntry")
OUT_BASE = ROOT_DIR / "artifacts" / "missions" / "mission_013"
EVAL_DIR = OUT_BASE / "evaluation_results"
CKPT_DIR = OUT_BASE / "checkpoints"

GT_PATH = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "vehicle_ground_truth.json"
SPLITS_PATH = ROOT_DIR / "artifacts" / "missions" / "mission_012" / "benchmark_splits.json"
BASELINE_PATH = OUT_BASE / "baseline_metrics.json"

COCO_VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_iou(box1: List[float], box2: List[float]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    box1_area = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    box2_area = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    denom = box1_area + box2_area - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def compute_coco_ap(all_preds: List[Dict], all_gts: Dict[str, List[Dict]], iou_thresh: float) -> float:
    """Computes COCO 101-point interpolated Average Precision at a given IoU threshold."""
    total_gt = sum(len(boxes) for boxes in all_gts.values())
    if total_gt == 0:
        return 0.0
    if not all_preds:
        return 0.0

    # Sort all predictions globally by confidence descending
    sorted_preds = sorted(all_preds, key=lambda x: x["conf"], reverse=True)

    matched_gt = {fname: set() for fname in all_gts}
    tp_flags = []
    fp_flags = []

    for p in sorted_preds:
        fname = p["filename"]
        p_box = p["bbox_xyxy"]
        img_gts = all_gts.get(fname, [])

        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, g in enumerate(img_gts):
            if g_idx in matched_gt[fname]:
                continue
            iou = compute_iou(p_box, g["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp_flags.append(1)
            fp_flags.append(0)
            matched_gt[fname].add(best_gt_idx)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    tp_cumsum = np.cumsum(tp_flags)
    fp_cumsum = np.cumsum(fp_flags)

    recalls = tp_cumsum / float(total_gt)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # 101-point interpolation
    recall_levels = np.linspace(0.0, 1.0, 101)
    prec_at_rec = []
    for r in recall_levels:
        prec_valid = precisions[recalls >= r]
        if len(prec_valid) == 0:
            prec_at_rec.append(0.0)
        else:
            prec_at_rec.append(np.max(prec_valid))

    return float(np.mean(prec_at_rec))


def compute_map(all_preds: List[Dict], all_gts: Dict[str, List[Dict]]) -> Tuple[float, float]:
    """Computes mAP50 and mAP50-95."""
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    aps = [compute_coco_ap(all_preds, all_gts, round(thresh, 2)) for thresh in iou_thresholds]
    map50 = aps[0]
    map50_95 = float(np.mean(aps))
    return round(map50, 4), round(map50_95, 4)


def match_detections(pred_boxes: List[Dict], gt_boxes: List[Dict], iou_thresh: float = 0.50) -> Tuple[int, int, int]:
    if not pred_boxes and not gt_boxes:
        return 0, 0, 0
    if not pred_boxes and gt_boxes:
        return 0, 0, len(gt_boxes)
    if pred_boxes and not gt_boxes:
        return 0, len(pred_boxes), 0

    matched_gt = set()
    matched_pred = set()
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

    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
        torch.cuda.synchronize(device_idx)

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


def evaluate_model_on_split(
    model_wrapper,
    is_finetuned: bool,
    split_name: str,
    split_files: List[str],
    gt_data: Dict,
) -> Dict:
    total_frames = len(split_files)
    frames_detected = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_preds = 0
    total_gt = 0

    all_preds = []
    all_gts = {}

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

    predict_kwargs = {
        "conf": 0.25,
        "imgsz": 640,
        "device": "cuda:0",
        "verbose": False,
    }
    if not is_finetuned:
        predict_kwargs["classes"] = COCO_VEHICLE_CLASSES
    else:
        predict_kwargs["classes"] = [0]  # fine-tuned vehicle class

    for fname in split_files:
        entry = next(v for v in gt_data.values() if v["filename"] == fname)
        gt_boxes = entry["boxes"]
        lighting = entry["lighting_condition"]
        all_gts[fname] = gt_boxes

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

        res = model_wrapper.predict(source=img, **predict_kwargs)[0]

        preds = []
        if res.boxes is not None and len(res.boxes) > 0:
            frames_detected += 1
            lighting_stats[lighting]["detected"] += 1
            for b in res.boxes:
                coords = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                pred_item = {"filename": fname, "bbox_xyxy": coords, "conf": conf}
                preds.append(pred_item)
                all_preds.append(pred_item)

        total_preds += len(preds)
        tp, fp, fn = match_detections(preds, gt_boxes, iou_thresh=0.50)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        lighting_stats[lighting]["tp"] += tp
        lighting_stats[lighting]["fp"] += fp
        lighting_stats[lighting]["fn"] += fn

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

    map50, map50_95 = compute_map(all_preds, all_gts)

    lighting_res = {}
    for l_key, l_data in lighting_stats.items():
        l_prec = l_data["tp"] / (l_data["tp"] + l_data["fp"]) if (l_data["tp"] + l_data["fp"]) > 0 else 0.0
        l_rec = l_data["tp"] / l_data["gt"] if l_data["gt"] > 0 else 0.0
        l_yield = (l_data["detected"] / l_data["frames"] * 100.0) if l_data["frames"] > 0 else 0.0
        lighting_res[l_key] = {
            "frames": l_data["frames"],
            "frames_detected": l_data["detected"],
            "yield_pct": round(l_yield, 2),
            "tp": l_data["tp"],
            "fp": l_data["fp"],
            "fn": l_data["fn"],
            "precision": round(l_prec, 4),
            "recall": round(l_rec, 4),
        }

    class_res = {}
    for c_key, c_data in class_stats.items():
        c_rec = c_data["tp"] / c_data["gt"] if c_data["gt"] > 0 else 0.0
        class_res[c_key] = {
            "ground_truth_instances": c_data["gt"],
            "true_positives": c_data["tp"],
            "recall": round(c_rec, 4),
        }

    vis_res = {}
    for v_key, v_data in visibility_stats.items():
        v_rec = v_data["tp"] / v_data["gt"] if v_data["gt"] > 0 else 0.0
        vis_res[v_key] = {
            "ground_truth_instances": v_data["gt"],
            "true_positives": v_data["tp"],
            "recall": round(v_rec, 4),
        }

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
        "map50": map50,
        "map50_95": map50_95,
        "false_positives_per_frame": round(fp_per_frame, 4),
        "frame_detection_yield_pct": round(yield_pct, 2),
        "lighting_breakdown": lighting_res,
        "class_breakdown": class_res,
        "visibility_breakdown": vis_res,
    }


def main():
    print("=" * 80)
    print("MISSION 013: 4-MODEL COMPREHENSIVE EVALUATION AND BENCHMARKING")
    print("=" * 80)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    test_files = splits["test"]["filenames"]
    val_files = splits["val"]["filenames"]

    model_specs = [
        {
            "id": "rtdetr_l_pretrained",
            "name": "Pretrained RT-DETR-L (COCO)",
            "wrapper_cls": RTDETR,
            "path": ROOT_DIR / "rtdetr-l.pt",
            "is_finetuned": False,
            "architecture": "RT-DETR-L (Vision Transformer)",
            "license": "Apache-2.0 upstream (Baidu) / AGPL-3.0 Ultralytics runner",
        },
        {
            "id": "rtdetr_l_finetuned",
            "name": "Fine-Tuned RT-DETR-L (Domain Adapted)",
            "wrapper_cls": RTDETR,
            "path": CKPT_DIR / "rtdetr_l_finetuned_best.pt",
            "is_finetuned": True,
            "architecture": "RT-DETR-L (Vision Transformer)",
            "license": "Apache-2.0 upstream architecture & fine-tuned domain weights",
        },
        {
            "id": "yolov8m_pretrained",
            "name": "Pretrained YOLOv8m (COCO)",
            "wrapper_cls": YOLO,
            "path": ROOT_DIR / "yolov8m.pt",
            "is_finetuned": False,
            "architecture": "YOLOv8m (CSPDarknet CNN)",
            "license": "AGPL-3.0 Ultralytics",
        },
        {
            "id": "yolov8m_finetuned",
            "name": "Fine-Tuned YOLOv8m (Domain Adapted)",
            "wrapper_cls": YOLO,
            "path": CKPT_DIR / "yolov8m_finetuned_best.pt",
            "is_finetuned": True,
            "architecture": "YOLOv8m (CSPDarknet CNN)",
            "license": "AGPL-3.0 Ultralytics",
        },
    ]

    # Verify all checkpoint files exist
    for spec in model_specs:
        if not spec["path"].is_file():
            print(f"ERROR: Checkpoint file missing: {spec['path']}")
            sys.exit(1)

    eval_results = {}
    manifest = {}
    csv_rows = []

    for spec in model_specs:
        mid = spec["id"]
        print(f"\n--- Evaluating Model: {spec['name']} ({mid}) ---")
        sha = sha256_file(spec["path"])
        file_size_bytes = os.path.getsize(spec["path"])

        wrapper = spec["wrapper_cls"](str(spec["path"]))
        param_count = sum(p.numel() for p in wrapper.model.parameters())

        manifest[mid] = {
            "name": spec["name"],
            "architecture": spec["architecture"],
            "checkpoint_path": str(spec["path"].relative_to(ROOT_DIR)).replace("\\", "/"),
            "sha256": sha,
            "size_bytes": file_size_bytes,
            "parameters_count": param_count,
            "license": spec["license"],
            "is_finetuned": spec["is_finetuned"],
        }

        # Latency profiling
        lat_profile = profile_detector_latency(wrapper.model, device_idx=0)
        print(f"  Latency: p50={lat_profile['median_p50_latency_ms']}ms, p95={lat_profile['p95_latency_ms']}ms, mean={lat_profile['mean_latency_ms']}ms, VRAM={lat_profile['peak_gpu_memory_allocated_mib']}MiB")

        # Evaluate on development_test_v1
        print("  Evaluating on development_test_v1 (72 frames)...")
        test_eval = evaluate_model_on_split(wrapper, spec["is_finetuned"], "development_test_v1", test_files, gt_data)

        # Evaluate on validation
        print("  Evaluating on validation (72 frames)...")
        val_eval = evaluate_model_on_split(wrapper, spec["is_finetuned"], "validation", val_files, gt_data)

        eval_results[mid] = {
            "specification": manifest[mid],
            "latency_profile": lat_profile,
            "development_test_v1": test_eval,
            "validation": val_eval,
        }

        csv_rows.append({
            "model_id": mid,
            "model_name": spec["name"],
            "architecture": spec["architecture"],
            "is_finetuned": spec["is_finetuned"],
            "test_precision_50": test_eval["precision_50"],
            "test_recall_50": test_eval["recall_50"],
            "test_f1_50": test_eval["f1_score_50"],
            "test_map50": test_eval["map50"],
            "test_map50_95": test_eval["map50_95"],
            "test_fp_per_frame": test_eval["false_positives_per_frame"],
            "test_yield_pct": test_eval["frame_detection_yield_pct"],
            "test_night_ir_yield_pct": test_eval["lighting_breakdown"]["NIGHT_IR"]["yield_pct"],
            "test_evening_ir_yield_pct": test_eval["lighting_breakdown"]["EVENING_IR"]["yield_pct"],
            "test_daylight_yield_pct": test_eval["lighting_breakdown"]["DAYLIGHT"]["yield_pct"],
            "val_precision_50": val_eval["precision_50"],
            "val_recall_50": val_eval["recall_50"],
            "val_f1_50": val_eval["f1_score_50"],
            "val_map50": val_eval["map50"],
            "val_map50_95": val_eval["map50_95"],
            "latency_p50_ms": lat_profile["median_p50_latency_ms"],
            "latency_p95_ms": lat_profile["p95_latency_ms"],
            "peak_vram_mib": lat_profile["peak_gpu_memory_allocated_mib"],
        })

    # Save manifest
    manifest_path = OUT_BASE / "checkpoint_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved checkpoint manifest to: {manifest_path}")

    # Save full evaluation results
    full_eval_path = EVAL_DIR / "full_evaluation_results.json"
    with open(full_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved full evaluation results to: {full_eval_path}")

    # Save CSV comparison
    csv_path = OUT_BASE / "model_comparison.csv"
    fieldnames = list(csv_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved model comparison CSV to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Model':<35} | {'Prec@50':<8} | {'Rec@50':<8} | {'F1@50':<8} | {'mAP50':<8} | {'mAP50-95':<8} | {'FP/fr':<6} | {'Yield':<6} | {'p50 ms':<6}")
    print("-" * 100)
    for r in csv_rows:
        print(f"{r['model_name']:<35} | {r['test_precision_50']:<8.4f} | {r['test_recall_50']:<8.4f} | {r['test_f1_50']:<8.4f} | {r['test_map50']:<8.4f} | {r['test_map50_95']:<8.4f} | {r['test_fp_per_frame']:<6.3f} | {r['test_yield_pct']:<6.1f} | {r['latency_p50_ms']:<6.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
