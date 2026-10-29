#!/usr/bin/env python3
"""Two-Way Evaluation of Egyptian License Plate OCR:
1. Oracle GT plate crops (upper bound / OCR module test).
2. End-to-End ALPR: Detector-produced crops from full vehicle images (headline metric).

Computes:
- Full-plate exact match accuracy (%)
- Character Error Rate (CER = edit_distance / len(gt))
- Normalized Levenshtein similarity
- Per-class confusion and error analysis for the worst-performing classes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES
from vehicle_intelligence.evaluation.ocr_eval import EgyptianPlateTextNormalizer
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def parse_character_boxes_from_yolo_result(yolo_result, conf_thresh: float = 0.25) -> List[CharacterBox]:
    """Extracts CharacterBox objects from YOLO character detector inference."""
    boxes = []
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return boxes

    for b in yolo_result.boxes:
        conf = float(b.conf[0])
        if conf < conf_thresh:
            continue
        cls_id = int(b.cls[0])
        # xywhn: normalized x_center, y_center, width, height
        xywhn = b.xywhn[0].tolist()
        boxes.append(
            CharacterBox(
                class_id=cls_id,
                x_center=xywhn[0],
                y_center=xywhn[1],
                width=xywhn[2],
                height=xywhn[3],
                confidence=conf,
            )
        )
    return boxes


def parse_gt_character_boxes_from_file(label_path: Path) -> List[CharacterBox]:
    """Loads ground-truth CharacterBox objects from YOLO label file."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x, y, w, h = [float(p) for p in parts[1:5]]
            boxes.append(CharacterBox(class_id=cls_id, x_center=x, y_center=y, width=w, height=h))
    return boxes


def evaluate_oracle_gt_crops(
    char_model: YOLO,
    char_dataset_dir: Path,
    conf_thresh: float = 0.25,
) -> dict:
    """Evaluates character recognition directly on ground-truth plate crops."""
    test_img_dir = char_dataset_dir / "images" / "test"
    test_lbl_dir = char_dataset_dir / "labels" / "test"

    img_files = sorted(list(test_img_dir.glob("*.png")))
    total = len(img_files)

    exact_matches = 0
    total_cer = 0.0
    total_edit_dist = 0
    total_gt_chars = 0
    per_plate_results = []
    confusion_errors = defaultdict(int)

    for img_path in img_files:
        stem = img_path.stem
        lbl_path = test_lbl_dir / f"{stem}.txt"

        gt_boxes = parse_gt_character_boxes_from_file(lbl_path)
        gt_full, gt_letters, gt_digits = EgyptianPlateReadingOrder.reconstruct_plate(gt_boxes)

        # Normalize spaces out for character-level matching
        gt_clean = gt_full.replace(" ", "")

        # Run inference
        res = char_model.predict(source=str(img_path), conf=conf_thresh, imgsz=320, verbose=False)[0]
        pred_boxes = parse_character_boxes_from_yolo_result(res, conf_thresh=conf_thresh)
        pred_full, pred_letters, pred_digits = EgyptianPlateReadingOrder.reconstruct_plate(pred_boxes)
        pred_clean = pred_full.replace(" ", "")

        dist = levenshtein_distance(gt_clean, pred_clean)
        gt_len = max(len(gt_clean), 1)
        cer = dist / gt_len

        is_exact = (gt_clean == pred_clean and len(gt_clean) > 0)
        if is_exact:
            exact_matches += 1

        total_cer += cer
        total_edit_dist += dist
        total_gt_chars += len(gt_clean)

        if not is_exact:
            confusion_errors[f"GT: '{gt_clean}' -> PRED: '{pred_clean}'"] += 1

        per_plate_results.append({
            "plate_id": stem,
            "gt_text": gt_full,
            "pred_text": pred_full,
            "exact_match": is_exact,
            "edit_distance": dist,
            "cer": round(cer, 4),
        })

    exact_match_acc = round((exact_matches / total * 100.0) if total else 0.0, 2)
    mean_cer = round((total_cer / total) if total else 0.0, 4)
    overall_char_acc = round(((total_gt_chars - total_edit_dist) / max(total_gt_chars, 1) * 100.0), 2)

    return {
        "evaluation_type": "Oracle Ground-Truth Plate Crops",
        "total_plates": total,
        "exact_matches": exact_matches,
        "exact_match_accuracy_pct": exact_match_acc,
        "mean_cer": mean_cer,
        "character_accuracy_pct": overall_char_acc,
        "worst_plate_errors": sorted(confusion_errors.items(), key=lambda x: x[1], reverse=True)[:10],
        "per_plate_results": per_plate_results,
    }


def evaluate_end_to_end_alpr(
    detector_model: YOLO,
    char_model: YOLO,
    ealpr_root: Path,
    test_manifest_path: Path,
    det_conf_thresh: float = 0.25,
    char_conf_thresh: float = 0.25,
) -> dict:
    """Evaluates full ALPR pipeline: Vehicle Image -> Detector -> Plate Crop -> OCR."""
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_vehicles = manifest["test_vehicles"]
    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    total_gt_plates = 0
    exact_matches = 0
    total_cer = 0.0
    total_edit_dist = 0
    total_gt_chars = 0
    detected_plates_count = 0
    missed_plates_count = 0

    confusion_errors = defaultdict(int)
    per_plate_results = []

    for stem in test_vehicles:
        img_path = v_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = v_dir / f"{stem}.jpeg"
        if not img_path.exists():
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Gather GT plates for this vehicle
        # Find matching character files e.g. stem_license_plate_1.txt
        gt_plate_files = sorted(list(cl_dir.glob(f"{stem}_license_plate_*.txt")))
        
        # Gather vehicle plate bounding boxes
        gt_boxes = []
        v_lbl = vl_dir / f"{stem}.txt"
        if v_lbl.exists():
            for line in v_lbl.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    bx, by, bw, bh = [float(p) for p in parts[1:5]]
                    gt_boxes.append(BoundingBox(
                        x1=(bx - bw/2) * w, y1=(by - bh/2) * h,
                        x2=(bx + bw/2) * w, y2=(by + bh/2) * h
                    ))

        # Run detector
        det_res = detector_model.predict(source=str(img_path), conf=det_conf_thresh, imgsz=640, verbose=False)[0]
        pred_plate_crops = []
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            for b in det_res.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop = im[y1:y2, x1:x2]
                    pred_plate_crops.append({
                        "box": BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        "crop": crop,
                        "conf": float(b.conf[0])
                    })

        # Match GT plate texts with crops
        for idx, cl_path in enumerate(gt_plate_files):
            total_gt_plates += 1
            gt_char_boxes = parse_gt_character_boxes_from_file(cl_path)
            gt_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(gt_char_boxes)
            gt_clean = gt_full.replace(" ", "")
            total_gt_chars += len(gt_clean)

            # Match with corresponding GT bounding box if available
            target_gt_box = gt_boxes[idx] if idx < len(gt_boxes) else (gt_boxes[0] if gt_boxes else None)

            matched_crop = None
            best_iou = 0.0
            if target_gt_box and pred_plate_crops:
                for p_crop in pred_plate_crops:
                    iou = target_gt_box.iou(p_crop["box"])
                    if iou > best_iou:
                        best_iou = iou
                        matched_crop = p_crop

            if matched_crop is not None and best_iou >= 0.50:
                detected_plates_count += 1
                # Run character recognition on the detector's crop
                c_res = char_model.predict(source=matched_crop["crop"], conf=char_conf_thresh, imgsz=320, verbose=False)[0]
                pred_char_boxes = parse_character_boxes_from_yolo_result(c_res, conf_thresh=char_conf_thresh)
                pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_char_boxes)
                pred_clean = pred_full.replace(" ", "")

                dist = levenshtein_distance(gt_clean, pred_clean)
                cer = dist / max(len(gt_clean), 1)
                is_exact = (gt_clean == pred_clean and len(gt_clean) > 0)
                if is_exact:
                    exact_matches += 1
                total_cer += cer
                total_edit_dist += dist

                if not is_exact:
                    confusion_errors[f"GT: '{gt_clean}' -> PRED: '{pred_clean}'"] += 1

                per_plate_results.append({
                    "vehicle_stem": stem,
                    "gt_text": gt_full,
                    "pred_text": pred_full,
                    "matched_iou": round(best_iou, 3),
                    "exact_match": is_exact,
                    "cer": round(cer, 4),
                })
            else:
                # Detection missed (False Negative)
                missed_plates_count += 1
                dist = len(gt_clean)
                total_cer += 1.0
                total_edit_dist += dist
                confusion_errors[f"GT: '{gt_clean}' -> MISSED_DETECTION"] += 1
                per_plate_results.append({
                    "vehicle_stem": stem,
                    "gt_text": gt_full,
                    "pred_text": "",
                    "matched_iou": round(best_iou, 3),
                    "exact_match": False,
                    "cer": 1.0,
                })

    exact_match_acc = round((exact_matches / total_gt_plates * 100.0) if total_gt_plates else 0.0, 2)
    mean_cer = round((total_cer / total_gt_plates) if total_gt_plates else 0.0, 4)
    overall_char_acc = round(((total_gt_chars - total_edit_dist) / max(total_gt_chars, 1) * 100.0), 2)

    return {
        "evaluation_type": "End-to-End ALPR Pipeline (Headline Metric)",
        "total_gt_plates": total_gt_plates,
        "plates_detected_by_detector": detected_plates_count,
        "plates_missed_by_detector": missed_plates_count,
        "exact_matches": exact_matches,
        "exact_match_accuracy_pct": exact_match_acc,
        "mean_cer": mean_cer,
        "character_accuracy_pct": max(0.0, overall_char_acc),
        "worst_plate_errors": sorted(confusion_errors.items(), key=lambda x: x[1], reverse=True)[:10],
        "per_plate_results": per_plate_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Egyptian License Plate Character Model.")
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--character-weights",
        type=Path,
        default=Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--char-dataset",
        type=Path,
        default=Path("artifacts/ealpr_characters_yolo_dataset"),
    )
    parser.add_argument(
        "--ealpr-root",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/cleaned_split_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation_ocr/ocr_two_way_report.json"),
    )
    args = parser.parse_args()

    print(f"Loading character model from {args.character_weights}...")
    char_model = YOLO(str(args.character_weights))

    print(f"Loading detector model from {args.detector_weights}...")
    detector_model = YOLO(str(args.detector_weights))

    print("\n--- 1. Evaluating on Oracle Ground-Truth Plate Crops ---")
    oracle_report = evaluate_oracle_gt_crops(char_model, args.char_dataset)

    print("\n--- 2. Evaluating End-to-End ALPR on Full Vehicle Images (Headline) ---")
    e2e_report = evaluate_end_to_end_alpr(
        detector_model=detector_model,
        char_model=char_model,
        ealpr_root=args.ealpr_root,
        test_manifest_path=args.manifest,
    )

    combined = {
        "oracle_gt_crops": oracle_report,
        "end_to_end_alpr": e2e_report,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EGYPTIAN ALPR CHARACTER RECOGNITION EVALUATION RESULTS")
    print("=" * 70)
    print("\n[Evaluation 1: Ground-Truth Plate Crops (Oracle OCR)]")
    print(f"  Total Plates Evaluated:     {oracle_report['total_plates']}")
    print(f"  Full-Plate Exact Match:     {oracle_report['exact_match_accuracy_pct']}% ({oracle_report['exact_matches']}/{oracle_report['total_plates']})")
    print(f"  Character Error Rate (CER): {oracle_report['mean_cer']}")
    print(f"  Character Accuracy:         {oracle_report['character_accuracy_pct']}%")

    print("\n[Evaluation 2: Detector-Produced Crops from Full Vehicles (HEADLINE METRIC)]")
    print(f"  Total GT Plates:            {e2e_report['total_gt_plates']}")
    print(f"  Plates Detected by Model:   {e2e_report['plates_detected_by_detector']}")
    print(f"  Full-Plate Exact Match:     {e2e_report['exact_match_accuracy_pct']}% ({e2e_report['exact_matches']}/{e2e_report['total_gt_plates']})")
    print(f"  Character Error Rate (CER): {e2e_report['mean_cer']}")
    print(f"  Character Accuracy:         {e2e_report['character_accuracy_pct']}%")
    print("\n" + "=" * 70)
    print(f"Saved complete evaluation report to: {args.output}\n")


if __name__ == "__main__":
    main()
