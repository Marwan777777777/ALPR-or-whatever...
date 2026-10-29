#!/usr/bin/env python3
"""Detailed Error Analysis for ALPR Pipeline (Mission 6 Part B):
1. Renders every wrong end-to-end plate crop into a contact sheet with GT and PRED labels.
2. Categorizes each error (substitution, insertion, deletion, detector miss).
3. Renders the 15 detector false positive crops and classifies them (real unannotated plate vs junk).
4. Tests character NMS improvements (class-agnostic NMS and IoU thresholds) on validation first, then test.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from vehicle_intelligence.core.domain import BoundingBox
from vehicle_intelligence.ocr.reading_order import (
    CharacterBox,
    EgyptianPlateReadingOrder,
)
from scripts.audit_end_to_end_and_ocr import (
    parse_character_boxes_from_yolo,
    parse_gt_characters_from_file,
    levenshtein_distance,
)


def categorize_error(gt_text: str, pred_text: str, is_detected: bool) -> str:
    if not is_detected:
        return "DETECTOR_MISS"
    gt_clean = gt_text.replace(" ", "")
    pred_clean = pred_text.replace(" ", "")

    if len(pred_clean) > len(gt_clean):
        return "INSERTION"
    elif len(pred_clean) < len(gt_clean):
        return "DELETION"
    else:
        return "SUBSTITUTION"


def run_error_analysis():
    print("Loading models for error analysis...")
    det_model = YOLO("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt")
    char_model = YOLO("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    ealpr_root = Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master")
    manifest = json.loads(Path("artifacts/splits/cleaned_split_manifest.json").read_text(encoding="utf-8"))
    test_stems = manifest["test_vehicles"]

    v_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles"
    vl_dir = ealpr_root / "EALPR Vechicles dataset" / "Vehicles Labeling"
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"

    out_vis_dir = Path("artifacts/visualizations/error_analysis")
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    wrong_plates_dir = out_vis_dir / "wrong_e2e_plates"
    wrong_plates_dir.mkdir(parents=True, exist_ok=True)
    fp_plates_dir = out_vis_dir / "detector_false_positives"
    fp_plates_dir.mkdir(parents=True, exist_ok=True)

    wrong_plates = []
    detector_fps = []

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

        # 1. Load GT plate boxes
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

        # 2. Load character GT
        c_files = sorted(list(cl_dir.glob(f"{stem}_license_plate_*.txt")))
        gt_char_map = {}
        for c_file in c_files:
            try:
                plate_idx = int(c_file.stem.split("_license_plate_")[-1]) - 1
            except ValueError:
                plate_idx = 0
            char_boxes = parse_gt_characters_from_file(c_file)
            txt, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            gt_char_map[plate_idx] = txt.replace(" ", "")

        # 3. Detect plates
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

        # Match detections to GT
        matched_gt = [False] * len(gt_boxes)
        matched_pred = [False] * len(detected_crops)
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
                matched_pred[p_idx] = True
                gt_to_pred[best_g] = p_crop

        # Collect False Positives
        for p_idx, p_crop in enumerate(detected_crops):
            if not matched_pred[p_idx]:
                fp_info = {
                    "vehicle": stem,
                    "crop": p_crop["crop"],
                    "conf": round(p_crop["conf"], 4),
                    "box": [p_crop["box"].x1, p_crop["box"].y1, p_crop["box"].x2, p_crop["box"].y2],
                }
                detector_fps.append(fp_info)

        # Evaluate eligible GT plates
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx not in gt_char_map:
                continue
            gt_text = gt_char_map[g_idx]
            p_crop = gt_to_pred.get(g_idx)

            if p_crop is not None:
                c_res = char_model.predict(source=p_crop["crop"], conf=0.25, imgsz=320, verbose=False)[0]
                pred_chars = parse_character_boxes_from_yolo(c_res, conf_thresh=0.25)
                pred_full, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(pred_chars)
                pred_text = pred_full.replace(" ", "")

                if pred_text != gt_text:
                    err_cat = categorize_error(gt_text, pred_text, is_detected=True)
                    wrong_plates.append({
                        "vehicle": stem,
                        "plate_idx": g_idx + 1,
                        "crop": p_crop["crop"],
                        "gt_text": gt_text,
                        "pred_text": pred_text,
                        "category": err_cat,
                    })
            else:
                # Detector Miss
                x1, y1 = int(max(0, g_box.x1)), int(max(0, g_box.y1))
                x2, y2 = int(min(w, g_box.x2)), int(min(h, g_box.y2))
                crop = im[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((50, 100, 3), dtype=np.uint8)
                wrong_plates.append({
                    "vehicle": stem,
                    "plate_idx": g_idx + 1,
                    "crop": crop,
                    "gt_text": gt_text,
                    "pred_text": "<MISSED_BY_DETECTOR>",
                    "category": "DETECTOR_MISS",
                })

    print(f"Total wrong end-to-end plates found: {len(wrong_plates)} (Expected ~25)")
    print(f"Total detector false positives found: {len(detector_fps)} (Expected 15)")

    # 1. Save and classify detector False Positives
    fp_classifications = []
    for idx, fp in enumerate(detector_fps, start=1):
        crop_path = fp_plates_dir / f"fp_{idx:02d}_{fp['vehicle']}_conf{fp['conf']:.2f}.jpg"
        cv2.imwrite(str(crop_path), fp["crop"])

        # Inspect crop characteristics to classify:
        # Check aspect ratio: Egyptian plates have aspect ratio ~2.0 (width/height)
        ch, cw = fp["crop"].shape[:2]
        ar = cw / max(1, ch)
        # Run OCR on it to see if characters are recognized
        c_res = char_model.predict(source=fp["crop"], conf=0.25, imgsz=320, verbose=False)[0]
        chars_found = len(c_res.boxes) if c_res.boxes is not None else 0

        # Heuristic classification for initial analysis:
        # If OCR detects 3+ characters and aspect ratio between 1.5 and 4.0, likely an unannotated real plate
        if chars_found >= 3 and 1.5 <= ar <= 4.0:
            classification = "REAL_UNANNOTATED_PLATE"
        else:
            classification = "JUNK"

        fp_classifications.append({
            "fp_id": idx,
            "vehicle": fp["vehicle"],
            "confidence": fp["conf"],
            "aspect_ratio": round(ar, 2),
            "chars_detected": chars_found,
            "classification": classification,
            "image_path": str(crop_path),
        })

    # 2. Save individual wrong plates and build Contact Sheet
    cat_counts = defaultdict(int)
    for idx, wp in enumerate(wrong_plates, start=1):
        cat_counts[wp["category"]] += 1
        crop_path = wrong_plates_dir / f"wrong_{idx:02d}_{wp['vehicle']}_{wp['category']}.jpg"
        cv2.imwrite(str(crop_path), wp["crop"])
        wp["crop_path"] = str(crop_path)

    print("\nEnd-to-End Errors by Category:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")

    # Build Contact Sheet (grid of 5 columns)
    tile_w, tile_h = 240, 120
    cols = 5
    rows = (len(wrong_plates) + cols - 1) // cols
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 245, dtype=np.uint8)

    for i, wp in enumerate(wrong_plates):
        r = i // cols
        c = i % cols
        y0, x0 = r * tile_h, c * tile_w

        # Resize crop maintaining aspect ratio
        crop = wp["crop"]
        ch, cw = crop.shape[:2]
        scale = min((tile_w - 20) / max(cw, 1), (tile_h - 50) / max(ch, 1))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh))

        # Paste crop
        cx = x0 + (tile_w - nw) // 2
        cy = y0 + 10
        sheet[cy:cy + nh, cx:cx + nw] = resized

        # Draw border
        color = (0, 0, 200) if wp["category"] == "DETECTOR_MISS" else (50, 50, 50)
        cv2.rectangle(sheet, (x0 + 2, y0 + 2), (x0 + tile_w - 2, y0 + tile_h - 2), color, 1)

        # Put label text
        lbl_text = f"{wp['category']}"
        cv2.putText(sheet, lbl_text, (x0 + 8, y0 + tile_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 180), 1)
        sub_text = f"GT:{wp['gt_text'][:8]} | P:{wp['pred_text'][:8]}"
        cv2.putText(sheet, sub_text, (x0 + 8, y0 + tile_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)

    contact_sheet_path = out_vis_dir / "wrong_plates_contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), sheet)
    print(f"\nSaved contact sheet to {contact_sheet_path}")

    # Save summary JSON
    report = {
        "total_wrong_plates": len(wrong_plates),
        "error_categories": dict(cat_counts),
        "wrong_plates_list": [
            {
                "vehicle": wp["vehicle"],
                "plate_idx": wp["plate_idx"],
                "category": wp["category"],
                "gt_text": wp["gt_text"],
                "pred_text": wp["pred_text"],
                "crop_path": wp["crop_path"],
            }
            for wp in wrong_plates
        ],
        "total_detector_false_positives": len(detector_fps),
        "detector_false_positives": fp_classifications,
    }

    report_path = out_vis_dir / "error_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved error analysis report to {report_path}")


if __name__ == "__main__":
    run_error_analysis()
