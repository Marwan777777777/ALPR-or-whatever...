#!/usr/bin/env python3
"""Mission 008: Targeted Improvement Experiments.

Controlled experiments based on Mission 007E/007F/008 diagnostics:
1. Experiment A: Character Detector NMS IoU Threshold Tuning (with agnostic_nms=True)
   Evaluates IoU thresholds [0.70, 0.60, 0.55, 0.50, 0.45] on the 300 validation crops,
   then verifies on the 81 gate triage crops.
2. Experiment B: Plate Crop Border Padding (0%, 4%, 6%, 8%, 10%)
   Evaluates impact of crop boundary expansion on E2E validation split and gate crops.
3. Experiment C: Plate Detector Resolution (640 vs 960 vs 1280) on 4K Gate Frames
   Evaluates detector recall on the 100 gate triage frames (including the 18 missed plates).

Outputs:
- artifacts/missions/mission_008/experiment_results.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from scripts.audit_end_to_end_and_ocr import (
    levenshtein_distance,
    parse_character_boxes_from_yolo,
    parse_gt_characters_from_file,
)
from scripts.score_labeled_triage import (
    align_and_classify_errors,
    normalize_plate_string,
    VALID_CHARACTERS,
    DOCUMENTED_OOV_LETTERS,
)
from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.ocr.reading_order import EgyptianPlateReadingOrder

sys.stdout.reconfigure(encoding="utf-8")


def run_experiment_a_char_iou(
    char_model: YOLO,
    char_val_dir: Path,
    gate_rows: List[Dict[str, str]],
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Experiment A: Evaluate Character Detector NMS IoU Thresholds with agnostic_nms=True."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Character Detector NMS IoU Threshold Tuning (agnostic_nms=True)")
    print("=" * 60)

    val_img_dir = char_val_dir / "images" / "val"
    val_lbl_dir = char_val_dir / "labels" / "val"
    val_imgs = sorted(list(val_img_dir.glob("*.png")))

    iou_candidates = [0.70, 0.60, 0.55, 0.50, 0.45]
    val_metrics = []

    for iou_t in iou_candidates:
        exact_matches = 0
        total_ed = 0
        total_gt_len = 0
        insertions = 0
        deletions = 0

        for img_p in val_imgs:
            lbl_p = val_lbl_dir / f"{img_p.stem}.txt"
            gt_boxes = parse_gt_characters_from_file(lbl_p)
            gt_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(gt_boxes)
            gt_norm = gt_full.replace(" ", "")

            im = cv2.imread(str(img_p))
            res = char_model.predict(
                source=im,
                conf=0.25,
                iou=iou_t,
                agnostic_nms=True,
                imgsz=320,
                device=device,
                verbose=False,
            )[0]
            boxes = parse_character_boxes_from_yolo(res, conf_thresh=0.25)
            pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes)
            pred_norm = pred_full.replace(" ", "")

            ed = levenshtein_distance(gt_norm, pred_norm)
            if ed == 0:
                exact_matches += 1
            elif len(pred_norm) > len(gt_norm):
                insertions += (len(pred_norm) - len(gt_norm))
            elif len(pred_norm) < len(gt_norm):
                deletions += (len(gt_norm) - len(pred_norm))

            total_ed += ed
            total_gt_len += len(gt_norm)

        acc = exact_matches / len(val_imgs) * 100.0
        cer = total_ed / max(1, total_gt_len)
        print(f"Val Split | iou={iou_t:.2f}: Exact={exact_matches}/{len(val_imgs)} ({acc:.2f}%), CER={cer:.4f}, Insertions={insertions}, Deletions={deletions}")
        val_metrics.append({
            "iou_thresh": iou_t,
            "exact_matches": exact_matches,
            "total_plates": len(val_imgs),
            "exact_match_pct": round(acc, 2),
            "total_edits": total_ed,
            "total_gt_characters": total_gt_len,
            "cer": round(cer, 4),
            "insertions": insertions,
            "deletions": deletions,
        })

    # Pick best on validation split
    best_val = max(val_metrics, key=lambda m: (m["exact_matches"], -m["cer"]))
    best_iou = best_val["iou_thresh"]
    print(f"Best IoU on Validation: {best_iou:.2f} (Exact: {best_val['exact_match_pct']}%, CER: {best_val['cer']})")

    # Evaluate on the 81 Gate Triage Crops comparing baseline (0.70) vs best_iou
    gate_results = {}
    for iou_t in [0.70, best_iou]:
        key = f"iou_{int(iou_t*100)}"
        exact_cnt = 0
        total_ed = 0
        total_gt_len = 0
        ins_cnt = 0

        eval_count = 0
        for idx, r in enumerate(gate_rows, start=1):
            crop_p = root_dir / "artifacts" / "triage" / "samples" / f"crop_{idx:03d}.jpg"
            if not crop_p.exists():
                continue
            status = r.get("status", "").strip().lower()
            unsupported = r.get("unsupported_chars", "").strip().lower() in ("yes", "true", "1")
            gt_l = r.get("true_letters", "").strip()
            gt_d = r.get("true_digits", "").strip()
            gt_norm = normalize_plate_string(f"{gt_l} {gt_d}")
            has_oov = any(ch in DOCUMENTED_OOV_LETTERS or ch not in VALID_CHARACTERS for ch in gt_l + gt_d)

            if status != "readable" or unsupported or has_oov:
                continue

            eval_count += 1
            im = cv2.imread(str(crop_p))
            res = char_model.predict(
                source=im,
                conf=0.25,
                iou=iou_t,
                agnostic_nms=True,
                imgsz=320,
                device=device,
                verbose=False,
            )[0]
            boxes = parse_character_boxes_from_yolo(res, conf_thresh=0.25)
            pred_text, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes)
            pred_norm = normalize_plate_string(pred_text)

            align = align_and_classify_errors(gt_norm, pred_norm)
            if align["edit_distance"] == 0:
                exact_cnt += 1
            total_ed += align["edit_distance"]
            total_gt_len += len(gt_norm)
            ins_cnt += align["insertions"]

        gate_results[key] = {
            "iou_thresh": iou_t,
            "exact_matches": exact_cnt,
            "denominator": eval_count,
            "exact_match_pct": round(exact_cnt / eval_count * 100.0, 2),
            "total_edits": total_ed,
            "cer": round(total_ed / max(1, total_gt_len), 4),
            "insertions": ins_cnt,
        }
        print(f"Gate Triage | iou={iou_t:.2f}: Exact={exact_cnt}/{eval_count} ({exact_cnt/eval_count*100:.2f}%), CER={total_ed/total_gt_len:.4f}, Insertions={ins_cnt}")

    return {
        "experiment_name": "Experiment A: Character NMS IoU Threshold Tuning",
        "validation_tuning_results": val_metrics,
        "selected_best_iou_on_val": best_iou,
        "gate_triage_comparison": gate_results,
    }


def run_experiment_b_crop_padding(
    det_model: YOLO,
    char_model: YOLO,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Experiment B: Evaluate Crop Boundary Padding (0%, 4%, 6%, 8%, 10%) on Gate Frames."""
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Plate Crop Border Padding on 100 Gate Frames")
    print("=" * 60)

    csv_p = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"
    with open(csv_p, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    padding_rates = [0.0, 0.04, 0.06, 0.08, 0.10]
    padding_results = []

    for pad_pct in padding_rates:
        exact_matches = 0
        total_ed = 0
        total_gt_len = 0
        detected_crops = 0

        eval_subset = []
        for idx, r in enumerate(rows, start=1):
            frame_p = root_dir / "artifacts" / "triage" / "samples" / f"frame_{idx:03d}.jpg"
            if not frame_p.exists():
                continue

            status = r.get("status", "").strip().lower()
            unsupported = r.get("unsupported_chars", "").strip().lower() in ("yes", "true", "1")
            gt_l = r.get("true_letters", "").strip()
            gt_d = r.get("true_digits", "").strip()
            gt_norm = normalize_plate_string(f"{gt_l} {gt_d}")
            has_oov = any(ch in DOCUMENTED_OOV_LETTERS or ch not in VALID_CHARACTERS for ch in gt_l + gt_d)

            # In-vocab readable with visible plate
            if status != "readable" or unsupported or has_oov:
                continue

            im = cv2.imread(str(frame_p))
            h, w = im.shape[:2]

            # Detect plate at 640
            det_res = det_model.predict(source=im, conf=0.25, imgsz=640, device=device, verbose=False)[0]
            if det_res.boxes is None or len(det_res.boxes) == 0:
                continue

            detected_crops += 1
            bx1, by1, bx2, by2 = [int(v) for v in det_res.boxes[0].xyxy[0].tolist()]

            # Apply symmetric padding
            bw, bh = bx2 - bx1, by2 - by1
            pad_w = int(bw * pad_pct)
            pad_h = int(bh * pad_pct)

            px1 = max(0, bx1 - pad_w)
            py1 = max(0, by1 - pad_h)
            px2 = min(w, bx2 + pad_w)
            py2 = min(h, by2 + pad_h)

            crop = im[py1:py2, px1:px2]

            # Character recognition with agnostic_nms=True
            c_res = char_model.predict(source=crop, conf=0.25, iou=0.70, agnostic_nms=True, imgsz=320, device=device, verbose=False)[0]
            char_boxes = parse_character_boxes_from_yolo(c_res, conf_thresh=0.25)
            pred_text, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            pred_norm = normalize_plate_string(pred_text)

            align = align_and_classify_errors(gt_norm, pred_norm)
            if align["edit_distance"] == 0:
                exact_matches += 1
            total_ed += align["edit_distance"]
            total_gt_len += len(gt_norm)

        acc = exact_matches / max(1, detected_crops) * 100.0
        cer = total_ed / max(1, total_gt_len)
        print(f"Crop Padding {pad_pct*100:4.1f}% | Detected={detected_crops}, Exact={exact_matches}/{detected_crops} ({acc:.2f}%), CER={cer:.4f}")
        padding_results.append({
            "padding_percent": round(pad_pct * 100.0, 1),
            "detected_plates": detected_crops,
            "exact_matches": exact_matches,
            "exact_match_pct": round(acc, 2),
            "total_edits": total_ed,
            "cer": round(cer, 4),
        })

    return {
        "experiment_name": "Experiment B: Plate Crop Border Padding",
        "results": padding_results,
    }


def run_experiment_c_detector_resolution(
    det_model: YOLO,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Experiment C: Evaluate Detector Resolution on the 100 Gate Frames (including 18 missed plates)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Plate Detector Resolution (640 vs 960 vs 1280) on Gate Frames")
    print("=" * 60)

    csv_p = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"
    with open(csv_p, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    visible_indices = [
        idx for idx, r in enumerate(rows, start=1)
        if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial")
    ]
    missed_visible_indices = [
        idx for idx, r in enumerate(rows, start=1)
        if r.get("plate_visible_to_eye", "").strip().lower() in ("yes", "partial")
        and r.get("crop_is_plate", "").strip().lower() == "no_crop"
    ]

    resolutions = [640, 960, 1280]
    res_metrics = []

    for sz in resolutions:
        detected_all = 0
        detected_visible = 0
        recovered_misses = []

        start_time = time.time()
        for idx in range(1, 101):
            frame_p = root_dir / "artifacts" / "triage" / "samples" / f"frame_{idx:03d}.jpg"
            if not frame_p.exists():
                continue
            im = cv2.imread(str(frame_p))

            res = det_model.predict(source=im, conf=0.25, imgsz=sz, device=device, verbose=False)[0]
            has_det = (res.boxes is not None and len(res.boxes) > 0)

            if has_det:
                detected_all += 1
                if idx in visible_indices:
                    detected_visible += 1
                if idx in missed_visible_indices:
                    max_conf = float(res.boxes.conf.max())
                    recovered_misses.append({"sample_id": f"sample_{idx:03d}", "row": idx, "conf": round(max_conf, 2)})

        elapsed = time.time() - start_time
        recall_pct = detected_visible / len(visible_indices) * 100.0
        fps = 100.0 / max(0.1, elapsed)

        print(f"imgsz={sz:4d} | Detected: {detected_all}/100, Visible Recall: {detected_visible}/{len(visible_indices)} ({recall_pct:.2f}%), Recovered Misses: {len(recovered_misses)}, FPS: {fps:.1f}")
        res_metrics.append({
            "imgsz": sz,
            "detected_all_frames": detected_all,
            "visible_plates_detected": detected_visible,
            "visible_plates_total": len(visible_indices),
            "detection_recall_pct": round(recall_pct, 2),
            "recovered_missed_plates_count": len(recovered_misses),
            "recovered_misses_detail": recovered_misses,
            "total_inference_time_seconds": round(elapsed, 2),
            "fps": round(fps, 1),
        })

    return {
        "experiment_name": "Experiment C: Plate Detector Input Resolution on 4K Gate Footage",
        "results": res_metrics,
    }


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running Mission 008 Controlled Experiments on device: {dev}")

    det_weights = root_dir / "artifacts" / "runs" / "detect" / "ealpr_cleaned_yolov8n" / "weights" / "best.pt"
    char_weights = root_dir / "artifacts" / "runs" / "detect_chars" / "ealpr_characters_yolov8n" / "weights" / "best.pt"
    char_val_dir = root_dir / "artifacts" / "ealpr_characters_yolo_dataset"
    csv_p = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"

    det_model = YOLO(str(det_weights))
    char_model = YOLO(str(char_weights))

    with open(csv_p, "r", encoding="utf-8-sig") as f:
        gate_rows = list(csv.DictReader(f))

    exp_a = run_experiment_a_char_iou(char_model, char_val_dir, gate_rows, device=dev)
    exp_b = run_experiment_b_crop_padding(det_model, char_model, device=dev)
    exp_c = run_experiment_c_detector_resolution(det_model, device=dev)

    all_experiments = {
        "metadata": {
            "mission": "MISSION 008",
            "date": "2026-10-30",
            "device": dev,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "experiment_a_char_nms_iou": exp_a,
        "experiment_b_crop_padding": exp_b,
        "experiment_c_detector_resolution": exp_c,
    }

    out_file = root_dir / "artifacts" / "missions" / "mission_008" / "experiment_results.json"
    out_file.write_text(json.dumps(all_experiments, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSuccessfully wrote all experiment results to: {out_file}")


if __name__ == "__main__":
    main()
