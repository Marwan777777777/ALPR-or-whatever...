#!/usr/bin/env python3
"""Gate Camera Triage Pipeline (Mission 007 Step 3).

Executes plate detector and character model on all images in gate footage:
- Saves annotated images to artifacts/triage/annotated/
- Saves plate crops to artifacts/triage/crops/
- Outputs artifacts/triage/triage.csv with per-image metrics
- Generates contact sheets for lowest-confidence detections and missed detections
- Computes lighting, width distributions, and review rates
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from scripts.audit_end_to_end_and_ocr import parse_character_boxes_from_yolo
from scripts.tune_confidence_and_coverage import compute_plate_confidence


def read_image_utf8(path: Path | str) -> Optional[np.ndarray]:
    """Safely reads images with UTF-8 / non-ASCII paths on Windows."""
    try:
        data = Path(path).read_bytes()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def write_image_utf8(path: Path | str, img: np.ndarray) -> bool:
    """Safely writes images with UTF-8 / non-ASCII paths on Windows."""
    try:
        ext = Path(path).suffix
        if not ext:
            ext = ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(buf.tobytes())
            return True
        return False
    except Exception as e:
        print(f"Error writing {path}: {e}")
        return False


def check_is_grayscale(im: np.ndarray, thresh: float = 4.0) -> bool:
    """Checks if image channels are nearly identical (IR or night camera mode)."""
    if len(im.shape) == 2 or im.shape[2] == 1:
        return True
    b = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    r = im[:, :, 2].astype(np.float32)
    diff_bg = np.mean(np.abs(b - g))
    diff_gr = np.mean(np.abs(g - r))
    return bool(diff_bg < thresh and diff_gr < thresh)


def compute_image_brightness(im: np.ndarray) -> float:
    """Computes mean grayscale brightness in [0, 255]."""
    if len(im.shape) == 2 or im.shape[2] == 1:
        return float(np.mean(im))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def create_contact_sheet(
    image_paths: List[Path],
    captions: List[str],
    output_path: Path,
    cols: int = 5,
    thumb_w: int = 320,
    thumb_h: int = 240,
    header_title: str = "Triage Contact Sheet",
) -> None:
    """Creates a high-contrast visual contact sheet of image thumbnails."""
    if not image_paths:
        return
    n = len(image_paths)
    rows = math.ceil(n / cols)
    margin = 10
    banner_h = 40
    caption_h = 45

    cell_w = thumb_w + margin
    cell_h = thumb_h + caption_h + margin

    total_w = cols * cell_w + margin
    total_h = rows * cell_h + banner_h + margin

    sheet = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    # Title
    draw.rectangle([0, 0, total_w, banner_h], fill=(15, 23, 42))
    draw.text((margin, 10), header_title, fill=(255, 255, 255))

    for idx, (img_path, caption) in enumerate(zip(image_paths, captions)):
        r = idx // cols
        c = idx % cols
        x = margin + c * cell_w
        y = banner_h + margin + r * cell_h

        # Read thumbnail
        im_cv = read_image_utf8(img_path)
        if im_cv is not None:
            im_pil = Image.fromarray(cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB))
            im_pil.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            # Center in thumbnail box
            pad_x = (thumb_w - im_pil.width) // 2
            pad_y = (thumb_h - im_pil.height) // 2
            sheet.paste(im_pil, (x + pad_x, y + pad_y))

        # Border
        draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline=(70, 70, 70), width=1)

        # Caption box
        cap_y = y + thumb_h + 2
        draw.rectangle([x, cap_y, x + thumb_w, cap_y + caption_h], fill=(20, 20, 20))
        lines = caption.split("\n")
        for l_idx, line in enumerate(lines[:3]):
            draw.text((x + 4, cap_y + 2 + l_idx * 14), line, fill=(220, 220, 220))

    sheet.save(str(output_path), quality=90)
    print(f"Saved contact sheet to {output_path} ({n} images, {rows}x{cols})")


def run_gate_triage(
    gate_dir: Path,
    output_dir: Path,
    det_weights: Path = Path("artifacts/runs/detect/ealpr_cleaned_yolov8n/weights/best.pt"),
    char_weights: Path = Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt"),
    device: str = "0",
) -> dict:
    annotated_dir = output_dir / "annotated"
    crops_dir = output_dir / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning gate folder: {gate_dir}...")
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in gate_dir.rglob("*") if f.suffix.lower() in valid_extensions])
    total_images = len(image_files)
    print(f"Found {total_images} images in gate folder.")

    if total_images == 0:
        raise ValueError(f"No images found in {gate_dir}")

    print(f"Loading plate detector from {det_weights}...")
    detector = YOLO(str(det_weights))
    print(f"Loading character detector from {char_weights}...")
    char_detector = YOLO(str(char_weights))

    csv_path = output_dir / "triage.csv"
    csv_rows = []

    plate_detected_count = 0
    plate_widths = []
    low_conf_results = []  # For contact sheet
    no_plate_results = []  # For contact sheet

    print("\nExecuting triage inference across all gate frames...")

    for idx, img_path in enumerate(image_files, 1):
        if idx % 50 == 0 or idx == total_images:
            print(f"  Processed {idx}/{total_images} frames...")

        im = read_image_utf8(img_path)
        if im is None:
            continue
        h, w = im.shape[:2]
        brightness = compute_image_brightness(im)
        is_gray = check_is_grayscale(im)

        # 1. Detect plates
        res = detector.predict(source=im, conf=0.25, imgsz=640, device=device, verbose=False)[0]

        detected_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                det_c = float(b.conf[0])
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                if bx2 > bx1 and by2 > by1:
                    detected_boxes.append({
                        "box": [bx1, by1, bx2, by2],
                        "conf": det_c,
                        "w": bx2 - bx1,
                        "h": by2 - by1,
                    })

        # Sort by confidence descending
        detected_boxes.sort(key=lambda x: x["conf"], reverse=True)
        num_plates = len(detected_boxes)
        plate_found = (num_plates > 0)

        annotated_im = im.copy()
        safe_stem = f"frame_{idx:04d}_{img_path.stem}"

        if plate_found:
            plate_detected_count += 1
            best_det = detected_boxes[0]
            bx1, by1, bx2, by2 = best_det["box"]
            p_w = best_det["w"]
            p_w_pct = round(p_w / w * 100.0, 2)
            det_conf = round(best_det["conf"], 4)
            plate_widths.append(p_w)

            # Crop best plate
            crop = im[by1:by2, bx1:bx2]
            crop_filename = f"{safe_stem}_crop.jpg"
            crop_path = crops_dir / crop_filename
            write_image_utf8(crop_path, crop)

            # 2. Run OCR on best plate crop (default agnostic_nms=False)
            c_res = char_detector.predict(
                source=crop, conf=0.25, iou=0.70, agnostic_nms=False, imgsz=320, verbose=False
            )[0]
            char_boxes = parse_character_boxes_from_yolo(c_res, conf_thresh=0.25)
            pred_text, _, _ = EgyptianPlateReadingOrder.reconstruct_plate(char_boxes)
            char_confs = [c.confidence for c in char_boxes if c.confidence is not None]
            plate_conf = compute_plate_confidence(det_conf, char_confs)

            needs_review = False
            review_reasons = []

            if plate_conf < 0.82:
                needs_review = True
                review_reasons.append(f"LOW_CONFIDENCE (score={plate_conf:.3f} < 0.82)")
            if p_w < 80:
                needs_review = True
                review_reasons.append(f"SMALL_PLATE_WIDTH ({p_w}px < 80px)")
            if len(char_boxes) < 4:
                needs_review = True
                review_reasons.append(f"FEW_CHARACTERS ({len(char_boxes)} < 4)")
            if num_plates > 1:
                needs_review = True
                review_reasons.append(f"MULTIPLE_PLATES ({num_plates})")
            if bx1 <= 5 or bx2 >= w - 5 or by1 <= 5 or by2 >= h - 5:
                needs_review = True
                review_reasons.append("PLATE_NEAR_FRAME_EDGE")

            # Draw annotation
            cv2.rectangle(annotated_im, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            label_text = f"Plate: {det_conf:.2f} | OCR: {len(char_boxes)} chars | RawConf: {plate_conf:.2f}"
            cv2.putText(annotated_im, label_text, (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if needs_review:
                cv2.putText(annotated_im, f"REVIEW: {'; '.join(review_reasons)}", (bx1, min(h - 10, by2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            low_conf_results.append({
                "image_path": img_path,
                "annotated_path": annotated_dir / f"{safe_stem}.jpg",
                "plate_conf": plate_conf,
                "det_conf": det_conf,
                "plate_w": p_w,
                "chars": len(char_boxes),
                "pred_text": pred_text,
                "reasons": review_reasons,
            })
        else:
            det_conf = 0.0
            p_w = 0.0
            p_w_pct = 0.0
            pred_text = ""
            plate_conf = 0.0
            needs_review = True
            review_reasons = ["NO_PLATE_DETECTED"]

            cv2.putText(annotated_im, "NO PLATE DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            no_plate_results.append({
                "image_path": img_path,
                "annotated_path": annotated_dir / f"{safe_stem}.jpg",
                "brightness": brightness,
                "is_gray": is_gray,
            })

        # Save annotated image
        annotated_path = annotated_dir / f"{safe_stem}.jpg"
        write_image_utf8(annotated_path, annotated_im)

        csv_rows.append({
            "file": img_path.name,
            "image_w": w,
            "image_h": h,
            "plate_found": plate_found,
            "num_plates": num_plates,
            "detector_conf": det_conf,
            "plate_w_px": round(p_w, 1),
            "plate_w_pct_of_image": p_w_pct,
            "mean_image_brightness": round(brightness, 1),
            "is_grayscale": is_gray,
            "predicted_text": pred_text,
            "plate_confidence": plate_conf,
            "needs_review": needs_review,
            "review_reasons": "; ".join(review_reasons),
        })

    # Write triage.csv
    fieldnames = [
        "file", "image_w", "image_h", "plate_found", "num_plates",
        "detector_conf", "plate_w_px", "plate_w_pct_of_image",
        "mean_image_brightness", "is_grayscale", "predicted_text",
        "plate_confidence", "needs_review", "review_reasons"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nWrote triage CSV to: {csv_path} ({len(csv_rows)} rows)")

    # 1. Detection Rate Overall
    det_rate_overall = round(plate_detected_count / total_images * 100.0, 2)

    # 2. Split by is_grayscale (night/IR vs color)
    gray_rows = [r for r in csv_rows if r["is_grayscale"]]
    color_rows = [r for r in csv_rows if not r["is_grayscale"]]
    gray_det_rate = round(sum(1 for r in gray_rows if r["plate_found"]) / max(1, len(gray_rows)) * 100.0, 2)
    color_det_rate = round(sum(1 for r in color_rows if r["plate_found"]) / max(1, len(color_rows)) * 100.0, 2)

    # 3. Split by brightness tercile
    brightnesses = [r["mean_image_brightness"] for r in csv_rows]
    b_t1, b_t2 = np.percentile(brightnesses, [33.33, 66.67])
    tercile1 = [r for r in csv_rows if r["mean_image_brightness"] <= b_t1]
    tercile2 = [r for r in csv_rows if b_t1 < r["mean_image_brightness"] <= b_t2]
    tercile3 = [r for r in csv_rows if r["mean_image_brightness"] > b_t2]

    t1_det_rate = round(sum(1 for r in tercile1 if r["plate_found"]) / max(1, len(tercile1)) * 100.0, 2)
    t2_det_rate = round(sum(1 for r in tercile2 if r["plate_found"]) / max(1, len(tercile2)) * 100.0, 2)
    t3_det_rate = round(sum(1 for r in tercile3 if r["plate_found"]) / max(1, len(tercile3)) * 100.0, 2)

    # 4. Plate width distribution in px
    if plate_widths:
        w_min = float(np.min(plate_widths))
        w_p10 = float(np.percentile(plate_widths, 10))
        w_med = float(np.median(plate_widths))
        w_p90 = float(np.percentile(plate_widths, 90))
        w_max = float(np.max(plate_widths))
        below_80_pct = round(sum(1 for w in plate_widths if w < 80) / len(plate_widths) * 100.0, 2)
    else:
        w_min = w_p10 = w_med = w_p90 = w_max = below_80_pct = 0.0

    # 5. Review Flag Rate
    needs_review_count = sum(1 for r in csv_rows if r["needs_review"])
    needs_review_pct = round(needs_review_count / total_images * 100.0, 2)

    # 6. Contact Sheet 1: 30 lowest-confidence results
    low_conf_results.sort(key=lambda x: x["plate_conf"])
    sheet1_items = low_conf_results[:30]
    sheet1_paths = [item["annotated_path"] for item in sheet1_items]
    sheet1_caps = [
        f"File: {item['image_path'].name[:22]}\nRawConf: {item['plate_conf']:.2f} | Det: {item['det_conf']:.2f}\nChars: {item['chars']} | W: {item['plate_w']}px"
        for item in sheet1_items
    ]
    sheet1_out = output_dir / "lowest_conf_contact_sheet.jpg"
    create_contact_sheet(sheet1_paths, sheet1_caps, sheet1_out, header_title="Top 30 Lowest-Confidence Predictions (Gate Triage)")

    # 7. Contact Sheet 2: 30 images with NO plate found
    sheet2_items = no_plate_results[:30]
    sheet2_paths = [item["annotated_path"] for item in sheet2_items]
    sheet2_caps = [
        f"File: {item['image_path'].name[:22]}\nBrightness: {item['brightness']:.1f}\nGrayscale/IR: {item['is_gray']}"
        for item in sheet2_items
    ]
    sheet2_out = output_dir / "no_plate_contact_sheet.jpg"
    create_contact_sheet(sheet2_paths, sheet2_caps, sheet2_out, header_title="Top 30 Missed Plate Frames (NO_PLATE_DETECTED)")

    summary = {
        "total_images": total_images,
        "detection_rate_overall_pct": det_rate_overall,
        "split_by_grayscale": {
            "grayscale_images_count": len(gray_rows),
            "grayscale_detection_rate_pct": gray_det_rate,
            "color_images_count": len(color_rows),
            "color_detection_rate_pct": color_det_rate,
        },
        "split_by_brightness_terciles": {
            "tercile_low_range": f"<= {b_t1:.1f}",
            "tercile_low_detection_rate_pct": t1_det_rate,
            "tercile_mid_range": f"{b_t1:.1f} - {b_t2:.1f}",
            "tercile_mid_detection_rate_pct": t2_det_rate,
            "tercile_high_range": f"> {b_t2:.1f}",
            "tercile_high_detection_rate_pct": t3_det_rate,
        },
        "plate_width_distribution_px": {
            "min": round(w_min, 1),
            "p10": round(w_p10, 1),
            "median": round(w_med, 1),
            "p90": round(w_p90, 1),
            "max": round(w_max, 1),
            "pct_below_80px": below_80_pct,
        },
        "needs_review_pct": needs_review_pct,
        "needs_review_count": needs_review_count,
        "contact_sheets": {
            "lowest_confidence": str(sheet1_out),
            "no_plate_found": str(sheet2_out),
        },
    }

    summary_path = output_dir / "triage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved triage summary to: {summary_path}")

    return summary


if __name__ == "__main__":
    gate_dir = Path(r"D:\TruckEntry")
    out_dir = Path("artifacts/triage")
    res = run_gate_triage(gate_dir=gate_dir, output_dir=out_dir)
    print("\n" + "=" * 60)
    print("TRIAGE EXECUTION COMPLETED")
    print("=" * 60)
    print(json.dumps(res, indent=2))
