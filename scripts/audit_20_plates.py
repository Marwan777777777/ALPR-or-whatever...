#!/usr/bin/env python3
"""Detailed 20-Plate Manual Reading-Order and Ground-Truth Audit.

Inspects 20 diverse test plates spanning:
- Varying digit counts (0, 1, 2, 3, 4)
- Varying letter counts (0, 1, 2, 3)
- Normal and low-resolution crops
- VIP / short-code plates
- Evaluates raw spatial box order vs. reconstructed GT string vs. model predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ultralytics import YOLO
from vehicle_intelligence.data.audit import EALPR_CHARACTER_CLASSES
from vehicle_intelligence.ocr.reading_order import CharacterBox, EgyptianPlateReadingOrder
from scripts.evaluate_character_model import parse_character_boxes_from_yolo_result


def audit_20_plates():
    ealpr_root = Path(r"C:\Users\MARWAN\plate_project\images_raw\EALPR-master")
    cl_dir = ealpr_root / "EALPR- LP characters dataset" / "Characters Labeling"
    p_dir = ealpr_root / "EALPR- Plates dataset"
    weights_path = Path("artifacts/runs/detect_chars/ealpr_characters_yolov8n/weights/best.pt")

    char_model = YOLO(str(weights_path))

    selected_20 = [
        "1241_license_plate_1",  # 0 digits, 3 letters
        "0990_license_plate_1",  # 1 digit, 1 letter (VIP)
        "0249_license_plate_1",  # 1 digit, 1 letter (VIP)
        "1664_license_plate_1",  # 1 digit, 2 letters
        "0663_license_plate_1",  # 1 digit, 2 letters
        "0501_license_plate_1",  # 1 digit, 3 letters
        "0312_license_plate_1",  # 1 digit, 3 letters
        "0045_license_plate_1",  # 2 digits, 1 letter (69x44 low-res)
        "0785_license_plate_1",  # 2 digits, 1 letter
        "0050_license_plate_1",  # 3 digits, 0 letters
        "1257_license_plate_2",  # 3 digits, 2 letters (73x47 low-res)
        "0040_license_plate_1",  # 3 digits, 2 letters
        "0538_license_plate_1",  # 3 digits, 3 letters
        "0483_license_plate_1",  # 3 digits, 3 letters
        "0541_license_plate_1",  # 4 digits, 0 letters
        "1665_license_plate_1",  # 4 digits, 2 letters
        "1420_license_plate_1",  # 4 digits, 2 letters
        "1893_license_plate_1",  # 4 digits, 3 letters
        "0731_license_plate_1",  # 4 digits, 3 letters
        "0559_license_plate_1",  # 4 digits, 3 letters
    ]

    audit_records = []

    print("\n" + "=" * 80)
    print("DETAILED 20-PLATE READING-ORDER & GROUND-TRUTH AUDIT")
    print("=" * 80 + "\n")

    for idx, p in enumerate(selected_20, 1):
        lbl_file = cl_dir / f"{p}.txt"
        im_file = p_dir / f"{p}.png"
        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        im_size = Image.open(im_file).size

        raw_chars = []
        for l in lines:
            parts = l.split()
            cid = int(parts[0])
            xc, yc, w, h = [float(x) for x in parts[1:]]
            ch = EALPR_CHARACTER_CLASSES.get(cid, "?")
            raw_chars.append({"cid": cid, "char": ch, "xc": xc, "yc": yc, "w": w, "h": h})

        # Purely Left-to-Right by x
        by_x = sorted(raw_chars, key=lambda c: c["xc"])
        by_x_str = "".join(c["char"] for c in by_x)

        # Digits vs Letters
        digits_boxes = [c for c in by_x if c["cid"] >= 17]
        letters_boxes = [c for c in by_x if c["cid"] < 17]

        # Reconstructed reading order (Letters RTL, Digits LTR)
        boxes = [CharacterBox(class_id=c["cid"], x_center=c["xc"], y_center=c["yc"], width=c["w"], height=c["h"]) for c in raw_chars]
        recon_full, recon_l, recon_d = EgyptianPlateReadingOrder.reconstruct_plate(boxes)

        # Run Model Prediction
        res = char_model.predict(source=str(im_file), conf=0.25, imgsz=320, verbose=False)[0]
        pred_boxes = parse_character_boxes_from_yolo_result(res, conf_thresh=0.25)
        pred_full, pred_l, pred_d = EgyptianPlateReadingOrder.reconstruct_plate(pred_boxes)

        is_exact = (recon_full.replace(" ", "") == pred_full.replace(" ", ""))

        record = {
            "index": idx,
            "plate_id": p,
            "image_size": list(im_size),
            "digit_count": len(digits_boxes),
            "letter_count": len(letters_boxes),
            "raw_ltr_boxes": by_x_str,
            "reconstructed_gt_string": recon_full,
            "predicted_string": pred_full,
            "exact_match": is_exact,
            "characters_detail": by_x,
        }
        audit_records.append(record)

        print(f"[{idx:02d}] {p} (Resolution: {im_size[0]}x{im_size[1]}, Digits: {len(digits_boxes)}, Letters: {len(letters_boxes)}):")
        print(f"     Raw LTR Box Order:    {by_x_str}")
        print(f"     Reconstructed GT:     {recon_full}")
        print(f"     Model Prediction:     {pred_full}  (Match: {is_exact})")
        print("     Box Coordinates:")
        for c in by_x:
            cat = "DIGIT " if c["cid"] >= 17 else "LETTER"
            print(f"       [{cat}] cid={c['cid']:2d} ('{c['char']}'): x_center={c['xc']:.3f}, y_center={c['yc']:.3f}, w={c['w']:.3f}, h={c['h']:.3f}")
        print("-" * 80)

    out_json = Path("artifacts/evaluation_ocr/manual_reading_order_audit_20_plates.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)
    print(f"\nAudit completed. Saved machine-readable report to: {out_json}")


if __name__ == "__main__":
    audit_20_plates()
