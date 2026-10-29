#!/usr/bin/env python3
"""Audited Evaluation of Egyptian Plate Detection and OCR (Mission 5 Section 2):
Part A: Full plate detection on all 323 test plates.
Part B: End-to-end ALPR on vehicle images (Detector -> Crop -> OCR) with one-to-one matching.
Part C: OCR-only evaluation on the 302 valid character-annotated plate crops.
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
from vehicle_intelligence.evaluation.detection_eval import (
    GroundTruthPlate,
    PlateDetectionEvaluator,
    PredictedPlate,
)
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein edit distance."""
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


def parse_character_boxes_from_yolo(res, conf_thresh: float = 0.25) -> List[CharacterBox]:
    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            conf = float(b.conf[0])
            if conf >= conf_thresh:
                cls_id = int(b.cls[0])
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


def parse_gt_characters_from_file(lbl_path: Path) -> List[CharacterBox]:
    if not lbl_path.is_file():
        return []
    boxes = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            xc, yc, w, h = [float(p) for p in parts[1:5]]
            boxes.append(CharacterBox(class_id=cls_id, x_center=xc, y_center=yc, width=w, height=h))
    return boxes


def run_full_plate_detection_eval(
    detector: YOLO,
    images_dir: Path,
    gt_json_path: Path,
    conf_thresh: float = 0.25,
) -> dict:
    """Part A: Evaluates plate detector on every valid ground truth plate in the test split."""
    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    ground_truths = [GroundTruthPlate(**g) for g in gt_data]

    valid_exts = {".jpg", ".jpeg", ".png"}
    img_files = sorted([f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])

    predictions: List[PredictedPlate] = []
    for img_file in img_files:
        stem = img_file.stem
        res = detector.predict(source=str(img_file), conf=conf_thresh, device=0, verbose=False)[0]
        h, w = res.orig_shape[:2]
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                score = float(b.conf[0])
                pred = PredictedPlate(
                    image_id=stem,
                    box=BoundingBox(
                        x1=max(0.0, min(1.0, bx1 / w)),
                        y1=max(0.0, min(1.0, by1 / h)),
                        x2=max(0.0, min(1.0, bx2 / w)),
                        y2=max(0.0, min(1.0, by2 / h)),
                    ),
                    confidence=round(score, 4),
                    class_name="license_plate",
                )
                predictions.append(pred)

    evaluator_50 = PlateDetectionEvaluator(iou_threshold=0.50)
    report_50 = evaluator_50.evaluate(ground_truths, predictions)

    evaluator_75 = PlateDetectionEvaluator(iou_threshold=0.75)
    report_75 = evaluator_75.evaluate(ground_truths, predictions)

    return {
        "total_test_images": len(img_files),
        "total_ground_truth_plates": len(ground_truths),
        "total_predictions": len(predictions),
        "confidence_threshold": conf_thresh,
        "metrics_iou_0_50": {
            "true_positives": report_50.true_positives,
            "false_positives": report_50.false_positives,
            "false_negatives": report_50.false_negatives,
            "precision": report_50.precision,
            "recall": report_50.recall,
            "f1_score": report_50.f1_score,
            "average_precision": report_50.average_precision,
            "mean_matched_iou": report_50.mean_matched_iou,
        },
        "metrics_iou_0_75": {
            "true_positives": report_75.true_positives,
            "false_positives": report_75.false_positives,
            "false_negatives": report_75.false_negatives,
            "precision": report_75.precision,
            "recall": report_75.recall,
            "f1_score": report_75.f1_score,
            "average_precision": report_75.average_precision,
            "mean_matched_iou": report_75.mean_matched_iou,
        },
    }


def run_ocr_only_eval(
    char_detector: YOLO,
    char_dataset_dir: Path,
    conf_thresh: float = 0.25,
) -> dict:
    """Part C: Evaluates OCR on valid character-annotated plate crops using ground-truth crops."""
    test_img_dir = char_dataset_dir / "images" / "test"
    test_lbl_dir = char_dataset_dir / "labels" / "test"

    img_files = sorted(list(test_img_dir.glob("*.png")))
    total_plates = len(img_files)

    exact_matches = 0
    total_edit_distance = 0
    total_gt_chars = 0
    total_pred_chars = 0
    plate_results = []

    for img_p in img_files:
        stem = img_p.stem
        lbl_p = test_lbl_dir / f"{stem}.txt"

        gt_boxes = parse_gt_characters_from_file(lbl_p)
        gt_str, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(gt_boxes)
        gt_norm = gt_str.replace(" ", "")

        res = char_detector.predict(source=str(img_p), conf=conf_thresh, imgsz=320, verbose=False)[0]
        pred_boxes = parse_character_boxes_from_yolo(res, conf_thresh=conf_thresh)
        pred_str, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_boxes)
        pred_norm = pred_str.replace(" ", "")

        dist = levenshtein_distance(gt_norm, pred_norm)
        is_exact = (gt_norm == pred_norm and len(gt_norm) > 0)
        if is_exact:
            exact_matches += 1

        total_edit_distance += dist
        total_gt_chars += len(gt_norm)
        total_pred_chars += len(pred_norm)

        plate_results.append({
            "plate_crop": stem,
            "gt_text": gt_str,
            "pred_text": pred_str,
            "gt_norm": gt_norm,
            "pred_norm": pred_norm,
            "exact_match": is_exact,
            "edit_distance": dist,
            "cer": round(dist / max(len(gt_norm), 1), 4),
        })

    exact_match_acc = round(exact_matches / total_plates * 100.0, 2) if total_plates else 0.0
    mean_cer = round(total_edit_distance / total_gt_chars, 4) if total_gt_chars else 0.0
    # Character accuracy defined as (total_gt_chars - edit_distance) / total_gt_chars
    char_acc = round(max(0.0, (total_gt_chars - total_edit_distance) / total_gt_chars * 100.0), 2) if total_gt_chars else 0.0

    return {
        "total_annotated_plates": total_plates,
        "exact_matches": exact_matches,
        "exact_match_accuracy_pct": exact_match_acc,
        "mean_cer": mean_cer,
        "character_accuracy_pct": char_acc,
        "total_gt_characters": total_gt_chars,
        "total_pred_characters": total_pred_chars,
        "total_edit_distance": total_edit_distance,
        "normalization_rules": "Strip whitespace; exact character code comparison.",
        "plate_results_sample": plate_results[:5],
    }


def run_end_to_end_eval(
    detector: YOLO,
    char_detector: YOLO,
    ealpr_root: Path,
    manifest_path: Path,
    det_conf_thresh: float = 0.25,
    char_conf_thresh: float = 0.25,
    match_iou_thresh: float = 0.50,
) -> dict:
    """Part B: Evaluates end-to-end ALPR: Full vehicle image -> Detector -> OCR -> Exact Match."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_stems = manifest["test_vehicles"]

    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    total_gt_plates = 0
    eligible_gt_plates = 0
    excluded_gt_plates = 0
    excluded_reasons = defaultdict(list)

    true_positive_detections = 0
    false_positive_detections = 0
    false_negative_detections = 0

    end_to_end_exact_matches = 0
    total_edit_distance = 0
    total_eligible_gt_chars = 0

    per_plate_records = []

    for stem in test_stems:
        img_path = v_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = v_dir / f"{stem}.jpeg"
        if not img_path.exists():
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        # 1. Gather all GT plate boxes on this vehicle from Vehicles Labeling
        v_lbl = vl_dir / f"{stem}.txt"
        gt_boxes = []
        if v_lbl.exists():
            for line in v_lbl.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                    gt_boxes.append(BoundingBox(
                        x1=(xc - bw / 2.0) * w,
                        y1=(yc - bh / 2.0) * h,
                        x2=(xc + bw / 2.0) * w,
                        y2=(yc + bh / 2.0) * h,
                    ))

        # 2. Gather character label files for this vehicle
        # Note: In EALPR, crops are named stem_license_plate_1.txt, stem_license_plate_2.txt
        c_files = sorted(list(cl_dir.glob(f"{stem}_license_plate_*.txt")))
        gt_char_map: Dict[int, str] = {}
        for c_file in c_files:
            # Parse plate index e.g. 1 from stem_license_plate_1.txt
            try:
                plate_idx = int(c_file.stem.split("_license_plate_")[-1]) - 1
            except ValueError:
                plate_idx = 0
            char_boxes = parse_gt_characters_from_file(c_file)
            txt, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            gt_char_map[plate_idx] = txt.replace(" ", "")

        # 3. Run detector on full vehicle image
        res = detector.predict(source=str(img_path), conf=det_conf_thresh, imgsz=640, device=0, verbose=False)[0]
        detected_crops = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    crop = im[by1:by2, bx1:bx2]
                    detected_crops.append({
                        "box": BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
                        "crop": crop,
                        "conf": float(b.conf[0]),
                    })

        # 4. One-to-one matching between GT plate boxes and detected crops
        matched_gt_flags = [False] * len(gt_boxes)
        matched_pred_flags = [False] * len(detected_crops)

        # Sort detected crops by confidence descending
        detected_crops.sort(key=lambda c: c["conf"], reverse=True)

        gt_to_pred_match: Dict[int, int] = {}
        for p_idx, p_crop in enumerate(detected_crops):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if not matched_gt_flags[g_idx]:
                    iou = p_crop["box"].iou(g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx
            if best_iou >= match_iou_thresh and best_g_idx >= 0:
                matched_gt_flags[best_g_idx] = True
                matched_pred_flags[p_idx] = True
                gt_to_pred_match[best_g_idx] = p_idx

        # Account for detection counts
        img_tp = sum(matched_gt_flags)
        img_fn = len(gt_boxes) - img_tp
        img_fp = len(detected_crops) - sum(matched_pred_flags)

        true_positive_detections += img_tp
        false_negative_detections += img_fn
        false_positive_detections += img_fp

        # 5. Evaluate end-to-end reading per ground truth plate
        for g_idx, g_box in enumerate(gt_boxes):
            total_gt_plates += 1
            has_gt_chars = g_idx in gt_char_map
            gt_text = gt_char_map.get(g_idx, "")

            if not has_gt_chars:
                excluded_gt_plates += 1
                excluded_reasons["missing_character_annotation"].append(f"{stem}_plate_{g_idx+1}")
                continue

            eligible_gt_plates += 1
            total_eligible_gt_chars += len(gt_text)

            p_idx = gt_to_pred_match.get(g_idx)
            if p_idx is not None:
                # Plate was detected -> Run OCR
                pred_crop_img = detected_crops[p_idx]["crop"]
                c_res = char_detector.predict(source=pred_crop_img, conf=char_conf_thresh, imgsz=320, verbose=False)[0]
                pred_chars = parse_character_boxes_from_yolo(c_res, conf_thresh=char_conf_thresh)
                pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_chars)
                pred_text = pred_full.replace(" ", "")

                dist = levenshtein_distance(gt_text, pred_text)
                is_exact = (gt_text == pred_text and len(gt_text) > 0)
                if is_exact:
                    end_to_end_exact_matches += 1
                total_edit_distance += dist

                per_plate_records.append({
                    "vehicle": stem,
                    "plate_index": g_idx + 1,
                    "status": "DETECTED",
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "exact_match": is_exact,
                    "edit_distance": dist,
                })
            else:
                # Detection missed -> End-to-end failure
                dist = len(gt_text)
                total_edit_distance += dist
                per_plate_records.append({
                    "vehicle": stem,
                    "plate_index": g_idx + 1,
                    "status": "MISSED_DETECTION",
                    "gt_text": gt_text,
                    "pred_text": "",
                    "exact_match": False,
                    "edit_distance": dist,
                })

    e2e_acc_eligible = round(end_to_end_exact_matches / eligible_gt_plates * 100.0, 2) if eligible_gt_plates else 0.0
    e2e_acc_all_gt = round(end_to_end_exact_matches / total_gt_plates * 100.0, 2) if total_gt_plates else 0.0
    mean_cer_eligible = round(total_edit_distance / total_eligible_gt_chars, 4) if total_eligible_gt_chars else 0.0

    return {
        "total_ground_truth_plates": total_gt_plates,
        "eligible_plates_with_character_gt": eligible_gt_plates,
        "excluded_plates_count": excluded_gt_plates,
        "excluded_reasons_summary": {k: len(v) for k, v in excluded_reasons.items()},
        "excluded_plate_samples": list(excluded_reasons["missing_character_annotation"])[:10],
        "plate_detection_summary": {
            "true_positives": true_positive_detections,
            "false_positives": false_positive_detections,
            "false_negatives": false_negative_detections,
            "precision": round(true_positive_detections / (true_positive_detections + false_positive_detections), 4) if (true_positive_detections + false_positive_detections) else 0.0,
            "recall": round(true_positive_detections / total_gt_plates, 4) if total_gt_plates else 0.0,
        },
        "end_to_end_reading": {
            "exact_matches": end_to_end_exact_matches,
            "exact_match_accuracy_on_eligible_gt_pct": e2e_acc_eligible,
            "exact_match_accuracy_on_all_gt_plates_pct": e2e_acc_all_gt,
            "mean_cer_on_eligible": mean_cer_eligible,
            "total_eligible_gt_characters": total_eligible_gt_chars,
            "total_edit_distance": total_edit_distance,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Audited evaluation of Egyptian plate detection and OCR.")
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--char-weights",
        type=Path,
        default=Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt"),
    )
    parser.add_argument(
        "--ealpr-root",
        type=Path,
        default=Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master"),
    )
    parser.add_argument(
        "--cleaned-dataset",
        type=Path,
        default=Path("artifacts/ealpr_yolo_dataset_cleaned"),
    )
    parser.add_argument(
        "--char-dataset",
        type=Path,
        default=Path("artifacts/ealpr_characters_yolo_dataset"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/splits/cleaned_split_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation_ocr"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    detector = YOLO(str(args.detector_weights))
    char_detector = YOLO(str(args.char_weights))

    print("\n--- Running Part A: Full Plate Detection Evaluation (All 323 Test Plates) ---")
    gt_json = Path("artifacts/splits/cleaned_ground_truth/test_ground_truth_plates.json")
    part_a = run_full_plate_detection_eval(
        detector=detector,
        images_dir=args.cleaned_dataset / "images" / "test",
        gt_json_path=gt_json,
    )
    print(f"Part A - TP: {part_a['metrics_iou_0_50']['true_positives']}, FP: {part_a['metrics_iou_0_50']['false_positives']}, FN: {part_a['metrics_iou_0_50']['false_negatives']}")
    print(f"Part A - Precision: {part_a['metrics_iou_0_50']['precision']}, Recall: {part_a['metrics_iou_0_50']['recall']}, F1: {part_a['metrics_iou_0_50']['f1_score']}, AP@50: {part_a['metrics_iou_0_50']['average_precision']}")

    print("\n--- Running Part C: OCR-Only Evaluation on Ground-Truth Crops (302 Plates) ---")
    part_c = run_ocr_only_eval(
        char_detector=char_detector,
        char_dataset_dir=args.char_dataset,
    )
    print(f"Part C - Exact Match Accuracy: {part_c['exact_match_accuracy_pct']}% ({part_c['exact_matches']}/{part_c['total_annotated_plates']})")
    print(f"Part C - Mean CER: {part_c['mean_cer']}, Character Accuracy: {part_c['character_accuracy_pct']}%")

    print("\n--- Running Part B: End-to-End ALPR Pipeline (Vehicle Image -> Detector -> OCR) ---")
    part_b = run_end_to_end_eval(
        detector=detector,
        char_detector=char_detector,
        ealpr_root=args.ealpr_root,
        manifest_path=args.manifest,
    )
    print(f"Part B - Total GT Plates in Vehicles: {part_b['total_ground_truth_plates']}")
    print(f"Part B - Eligible GT Plates with Character Annotations: {part_b['eligible_plates_with_character_gt']}")
    print(f"Part B - Excluded Plates (Missing Character Labels): {part_b['excluded_plates_count']}")
    print(f"Part B - End-to-End Exact Match on Eligible Plates: {part_b['end_to_end_reading']['exact_match_accuracy_on_eligible_gt_pct']}% ({part_b['end_to_end_reading']['exact_matches']}/{part_b['eligible_plates_with_character_gt']})")
    print(f"Part B - End-to-End Exact Match over ALL GT Plates (incl. unannotated): {part_b['end_to_end_reading']['exact_match_accuracy_on_all_gt_plates_pct']}%")
    print(f"Part B - Detector Performance in Pipeline: TP={part_b['plate_detection_summary']['true_positives']}, FP={part_b['plate_detection_summary']['false_positives']}, FN={part_b['plate_detection_summary']['false_negatives']}")

    full_report = {
        "part_a_full_plate_detection": part_a,
        "part_b_end_to_end_alpr": part_b,
        "part_c_ocr_only": part_c,
    }

    out_json = args.output_dir / "audited_end_to_end_report.json"
    out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved audited report to {out_json}")


if __name__ == "__main__":
    main()
