#!/usr/bin/env python3
"""Plate-level Confidence and Coverage vs Accuracy Calibration (Mission 6 Part C).
1. Defines composite plate confidence C = det_conf * (prod(char_confs))^(1/k).
2. Computes coverage-vs-accuracy curve on Validation split.
3. Chooses threshold tau on Validation achieving >= 99.0% exact-match accuracy on accepted plates.
4. Reports coverage and accuracy once on Test split.
5. Saves curve and threshold data for integration into needs_review.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.ocr.reading_order import EgyptianPlateReadingOrder
from scripts.audit_end_to_end_and_ocr import (
    parse_character_boxes_from_yolo,
    parse_gt_characters_from_file,
    levenshtein_distance,
)


def compute_plate_confidence(det_conf: float, char_confs: List[float]) -> float:
    """Computes composite plate confidence from detector confidence and character confidences."""
    if not char_confs:
        return 0.0
    # Geometric mean of character confidences * detector confidence
    log_sum = sum(math.log(max(1e-5, c)) for c in char_confs)
    geom_mean = math.exp(log_sum / len(char_confs))
    return round(float(det_conf * geom_mean), 4)


def extract_split_samples(split_name: str, det_model: YOLO, char_model: YOLO) -> List[dict]:
    ealpr_root = Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master")
    manifest = json.loads(Path("artifacts/splits/cleaned_split_manifest.json").read_text(encoding="utf-8"))
    stems = manifest[f"{split_name}_vehicles"]

    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    samples = []

    for stem in stems:
        img_path = v_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = v_dir / f"{stem}.jpeg"
        if not img_path.exists():
            continue

        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

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

        c_files = sorted(list(cl_dir.glob(f"{stem}_license_plate_*.txt")))
        gt_char_map = {}
        for c_file in c_files:
            try:
                pidx = int(c_file.stem.split("_license_plate_")[-1]) - 1
            except ValueError:
                pidx = 0
            char_boxes = parse_gt_characters_from_file(c_file)
            txt, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            gt_char_map[pidx] = txt.replace(" ", "")

        res = det_model.predict(source=str(img_path), conf=0.25, imgsz=640, device=0, verbose=False)[0]
        detected_crops = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    detected_crops.append({
                        "box": BoundingBox(x1=bx1, y1=by1, x2=bx2, y2=by2),
                        "crop": im[by1:by2, bx1:bx2],
                        "conf": float(b.conf[0]),
                    })

        matched_gt = [False] * len(gt_boxes)
        detected_crops.sort(key=lambda c: c["conf"], reverse=True)
        gt_to_pred = {}
        for p_idx, p_crop in enumerate(detected_crops):
            best_iou = 0.0
            best_g = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if not matched_gt[g_idx]:
                    iou = p_crop["box"].iou(g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_g = g_idx
            if best_iou >= 0.50 and best_g >= 0:
                matched_gt[best_g] = True
                gt_to_pred[best_g] = p_crop

        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx not in gt_char_map:
                continue
            gt_text = gt_char_map[g_idx]
            p_crop = gt_to_pred.get(g_idx)

            if p_crop is not None:
                # Use tuned agnostic_nms=True
                c_res = char_model.predict(
                    source=p_crop["crop"], conf=0.25, iou=0.70, agnostic_nms=True, imgsz=320, verbose=False
                )[0]
                pred_chars = parse_character_boxes_from_yolo(c_res, conf_thresh=0.25)
                pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_chars)
                pred_text = pred_full.replace(" ", "")

                char_confs = [c.confidence for c in pred_chars if c.confidence is not None]
                plate_conf = compute_plate_confidence(p_crop["conf"], char_confs)
                is_exact = (pred_text == gt_text and len(gt_text) > 0)

                samples.append({
                    "vehicle": stem,
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "det_conf": p_crop["conf"],
                    "char_confs": char_confs,
                    "plate_conf": plate_conf,
                    "exact_match": is_exact,
                })
            else:
                samples.append({
                    "vehicle": stem,
                    "gt_text": gt_text,
                    "pred_text": "",
                    "det_conf": 0.0,
                    "char_confs": [],
                    "plate_conf": 0.0,
                    "exact_match": False,
                })

    return samples


def compute_curve(samples: List[dict]) -> Tuple[List[dict], float]:
    """Computes coverage vs accuracy for thresholds in [0.0, 1.0]."""
    total = len(samples)
    thresholds = [round(t, 2) for t in np.linspace(0.40, 0.95, 56)]
    curve = []
    chosen_tau = 0.90
    found_tau = False

    for tau in thresholds:
        accepted = [s for s in samples if s["plate_conf"] >= tau]
        n_acc = len(accepted)
        if n_acc == 0:
            continue
        n_correct = sum(1 for s in accepted if s["exact_match"])
        cov = n_acc / total
        acc = n_correct / n_acc

        curve.append({
            "threshold": tau,
            "accepted_count": n_acc,
            "total_count": total,
            "coverage_pct": round(cov * 100.0, 2),
            "accuracy_pct": round(acc * 100.0, 2),
        })

        if acc >= 0.990 and not found_tau:
            chosen_tau = tau
            found_tau = True

    return curve, chosen_tau


def main():
    print("Loading models for confidence calibration...")
    det_model = YOLO("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt")
    char_model = YOLO("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    print("\nExtracting Validation split predictions...")
    val_samples = extract_split_samples("val", det_model, char_model)
    print(f"Loaded {len(val_samples)} validation plate evaluations.")

    val_curve, chosen_tau = compute_curve(val_samples)
    val_point = next(p for p in val_curve if p["threshold"] == chosen_tau)

    print(f"\nValidation Calibration Result:")
    print(f"  Target: >= 99.0% Exact Match on accepted plates")
    print(f"  Selected Threshold (tau): {chosen_tau}")
    print(f"  Coverage on Val:          {val_point['coverage_pct']}% ({val_point['accepted_count']}/{val_point['total_count']})")
    print(f"  Accuracy on Val Accepted: {val_point['accuracy_pct']}%")

    print("\nReporting ONCE on Test split using calibrated threshold tau = {:.2f}...".format(chosen_tau))
    test_samples = extract_split_samples("test", det_model, char_model)
    test_curve, _ = compute_curve(test_samples)
    test_point = next(p for p in test_curve if p["threshold"] == chosen_tau)

    print(f"Test Result at tau = {chosen_tau}:")
    print(f"  Coverage on Test:          {test_point['coverage_pct']}% ({test_point['accepted_count']}/{test_point['total_count']})")
    print(f"  Accuracy on Test Accepted: {test_point['accuracy_pct']}%")

    out_path = Path("artifacts/evaluation_ocr/confidence_coverage_calibration.json")
    out_path.write_text(json.dumps({
        "selected_threshold_on_val": chosen_tau,
        "val_operating_point": val_point,
        "test_evaluation_point": test_point,
        "val_curve": val_curve,
        "test_curve": test_curve,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved confidence calibration data to {out_path}")


if __name__ == "__main__":
    main()
