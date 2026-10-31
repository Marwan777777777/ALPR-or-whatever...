#!/usr/bin/env python3
"""Export per-sample comparison of all 77 OCR crops between IoU 0.70 and 0.45."""

import csv
import json
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

# Add repo root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.ocr.reading_order import EgyptianPlateReadingOrder
from scripts.audit_end_to_end_and_ocr import parse_character_boxes_from_yolo
from scripts.evaluate_gate_crops_nms import VALID_CHARACTERS, DOCUMENTED_OOV_LETTERS
from scripts.score_labeled_triage import normalize_plate_string, align_and_classify_errors

sys.stdout.reconfigure(encoding="utf-8")

def export_iou_comparison():
    char_model = YOLO("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")
    csv_p = root_dir / "artifacts" / "triage" / "completed_annotations_007e.csv"

    with open(csv_p, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_records = []
    
    for idx, r in enumerate(rows, start=1):
        status = r.get("status", "").strip().lower()
        crop = r.get("crop_is_plate", "").strip().lower()
        unsupported = r.get("unsupported_chars", "").strip().lower() in ("yes", "true", "1")
        letters = r.get("true_letters", "").strip()
        digits = r.get("true_digits", "").strip()
        has_oov = any(ch in DOCUMENTED_OOV_LETTERS or ch not in VALID_CHARACTERS for ch in letters + digits)

        if not (status == "readable" and crop == "yes" and not unsupported and not has_oov):
            continue  # Denominator filter

        gt_raw = f"{letters} {digits}".strip()
        gt_norm = normalize_plate_string(gt_raw)
        crop_path = root_dir / "artifacts" / "triage" / "samples" / f"crop_{idx:03d}.jpg"
        if not crop_path.exists():
            continue

        im = cv2.imread(str(crop_path))

        # Predict with IoU 0.70
        res_70 = char_model.predict(source=im, conf=0.25, iou=0.70, agnostic_nms=True, imgsz=320, device="cuda:0", verbose=False)[0]
        boxes_70 = parse_character_boxes_from_yolo(res_70, conf_thresh=0.25)
        pred_70, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes_70)
        pred_70_norm = normalize_plate_string(pred_70)
        align_70 = align_and_classify_errors(gt_norm, pred_70_norm)
        exact_70 = (pred_70_norm == gt_norm and len(gt_norm) > 0)

        # Predict with IoU 0.45
        res_45 = char_model.predict(source=im, conf=0.25, iou=0.45, agnostic_nms=True, imgsz=320, device="cuda:0", verbose=False)[0]
        boxes_45 = parse_character_boxes_from_yolo(res_45, conf_thresh=0.25)
        pred_45, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(boxes_45)
        pred_45_norm = normalize_plate_string(pred_45)
        align_45 = align_and_classify_errors(gt_norm, pred_45_norm)
        exact_45 = (pred_45_norm == gt_norm and len(gt_norm) > 0)

        if exact_70 and not exact_45:
            classification = "REGRESSION"
        elif not exact_70 and exact_45:
            classification = "IMPROVEMENT"
        elif exact_70 and exact_45:
            classification = "UNCHANGED_EXACT"
        else:
            classification = "UNCHANGED_ERROR"

        out_records.append({
            "row": idx,
            "sample_id": f"sample_{idx:03d}",
            "ground_truth": gt_norm,
            "pred_iou_70": pred_70_norm,
            "exact_iou_70": exact_70,
            "edit_distance_iou_70": align_70["edit_distance"],
            "pred_iou_45": pred_45_norm,
            "exact_iou_45": exact_45,
            "edit_distance_iou_45": align_45["edit_distance"],
            "delta_edit_distance": align_45["edit_distance"] - align_70["edit_distance"],
            "classification": classification,
        })

    out_csv = root_dir / "artifacts" / "missions" / "mission_009" / "iou_70_vs_45_per_sample.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_records[0].keys()))
        writer.writeheader()
        writer.writerows(out_records)

    print(f"Exported {len(out_records)} samples to {out_csv}")
    print(f"Classification breakdown:")
    for cat in ["UNCHANGED_EXACT", "IMPROVEMENT", "UNCHANGED_ERROR", "REGRESSION"]:
        cnt = sum(1 for r in out_records if r["classification"] == cat)
        print(f"  {cat}: {cnt}")

if __name__ == "__main__":
    export_iou_comparison()
